/* 「站點供需型態分析」頁專用：整日 +30 分鐘 MME 預測彙整（供舊版三欄）。
 *
 * 原則（僅服務此頁，不動其他頁面 DOM / 資料流 / API 格式）：
 *  - 數字不得取自 appData.js。對回放日期的每個半小時時段，透過既有後端 API（/api/replay）
 *    取得既有 SageMaker MME 的真實 +30 分鐘預測，再按 stationKey 彙整整日結果並快取。
 *  - 限制同時請求數（並行上限）＋整日快取，避免每次搜尋/排序都重打 Endpoint。
 *  - +30 分鐘的「目標時間」＝輸入時段 + 30 分（跨日以 24 小時取模），不把輸入時間當預測時間。
 *  - 定義：
 *      平均滿載率 = 各有效時段『預測可借車比例』平均。
 *      最忙時段   = 相鄰有效預測時段之『預測車數變動』最大者（顯示其 +30 目標時間）；
 *                   欄位提示註明這是預測車數變動，非真實租借交易量。
 *      風險時數   = 模型 +30 缺車或滿站預警的時段數 × 0.5 小時（用後端回傳門檻）。
 *  - 有效時段不足者標「統計資料不足（有效 N 時段）」，不補 0 / 40% / 08:00 / 舊版值。
 */
(function () {
    // 預設本機開發指向 http://127.0.0.1:8000（見 liveMode.js 說明）；可用 window.YOUBIKE_API_BASE 覆寫為 ''（同源）。
    const API_BASE = (typeof window.YOUBIKE_API_BASE === 'string' && window.YOUBIKE_API_BASE)
        ? window.YOUBIKE_API_BASE.replace(/\/$/, '')
        : 'http://127.0.0.1:8000';
    const REPLAY_DATE = () => (typeof window.YOUBIKE_REPLAY_DATE === 'string' && window.YOUBIKE_REPLAY_DATE)
        ? window.YOUBIKE_REPLAY_DATE : '2026-06-29';

    const CONCURRENCY = 4;         // 對 Endpoint 的同時請求上限
    const MIN_VALID_SLOTS = 4;     // 少於此有效時段數 → 統計資料不足
    const SLOT_MIN = 30;           // 每格 30 分鐘

    // 整日快取：date -> { statsByKey: Map, meta:{thresholds,...}, builtAt }
    const _dayCache = new Map();
    let _loadingPromise = null;    // 同一日期只跑一次
    let _loadingDate = null;

    function slots48() {
        if (typeof overviewTimelineSlots !== 'undefined' && Array.isArray(overviewTimelineSlots) && overviewTimelineSlots.length) {
            return overviewTimelineSlots.slice();
        }
        return Array.from({ length: 48 }, (_, i) => {
            const h = String(Math.floor(i / 2)).padStart(2, '0');
            return `${h}:${i % 2 === 0 ? '00' : '30'}`;
        });
    }

    // 輸入時段（HH:MM）+ 30 分鐘 的目標時間（跨日以 24h 取模）。回傳 {time:'HH:MM', dayOffset:0|1}
    function plus30Target(slot) {
        const [h, m] = slot.split(':').map(Number);
        const total = h * 60 + m + 30;
        const dayOffset = Math.floor(total / (24 * 60));
        const wrapped = total % (24 * 60);
        const th = String(Math.floor(wrapped / 60)).padStart(2, '0');
        const tm = String(wrapped % 60).padStart(2, '0');
        return { time: `${th}:${tm}`, dayOffset };
    }

    async function fetchSlot(date, slot) {
        const controller = new AbortController();
        const timer = setTimeout(() => controller.abort(), 180000);
        try {
            const resp = await fetch(`${API_BASE}/api/replay?date=${date}&time=${slot}`, { signal: controller.signal });
            if (!resp.ok) throw new Error('replay ' + slot + ' HTTP ' + resp.status);
            return await resp.json();
        } finally {
            clearTimeout(timer);
        }
    }

    // 並行上限的批次抓取，逐格回報進度。
    async function fetchAllSlots(date, slots, onProgress) {
        const results = new Array(slots.length);
        let done = 0;
        let next = 0;
        async function worker() {
            while (true) {
                const i = next++;
                if (i >= slots.length) return;
                try {
                    results[i] = { slot: slots[i], body: await fetchSlot(date, slots[i]) };
                } catch (e) {
                    results[i] = { slot: slots[i], error: (e && e.message) || String(e) };
                }
                done += 1;
                if (onProgress) onProgress(done, slots.length);
            }
        }
        const workers = [];
        for (let k = 0; k < Math.min(CONCURRENCY, slots.length); k++) workers.push(worker());
        await Promise.all(workers);
        return results;
    }

    // 由整日各時段回應，按 stationKey 彙整。
    function aggregate(slotResults) {
        // 門檻：取任一成功時段的 warningThresholds（後端回傳）。
        let thresholds = null;
        for (const r of slotResults) {
            if (r && r.body && r.body.warningThresholds) { thresholds = r.body.warningThresholds; break; }
        }
        // perKey: stationKey -> { name, district, cluster, total, series:[{slot, target, valid, bikeRatio, bikes, alerted}] }
        const perKey = new Map();
        for (const r of slotResults) {
            if (!r || !r.body || !Array.isArray(r.body.stations)) continue;
            const target = plus30Target(r.slot);
            for (const st of r.body.stations) {
                const key = st.stationKey;
                if (!key) continue;
                let rec = perKey.get(key);
                if (!rec) {
                    rec = { name: st.name, district: st.district, cluster: st.cluster,
                            total: st.total, series: [] };
                    perKey.set(key, rec);
                }
                const p30 = (st.predictions || {})['30'];
                const ok = st.dataStatus === 'ok' && p30 && Number.isFinite(p30.bikeRatio);
                // +30 缺車 / 滿站預警：讀後端 authoritative alerts（horizon 30、非 immediate）。
                // 分開記錄缺車與滿站，供「連續模型預警」區分重複缺車 vs 重複滿車。
                let shortageAlert = false, fullAlert = false;
                for (const a of (st.alerts || [])) {
                    if (!a.immediate && a.horizon === 30) {
                        if (a.kind === 'shortage') shortageAlert = true;
                        else if (a.kind === 'full') fullAlert = true;
                    }
                }
                rec.series.push({
                    slot: r.slot, target: target.time, dayOffset: target.dayOffset,
                    valid: !!ok,
                    bikeRatio: ok ? p30.bikeRatio : null,
                    bikes: (ok && Number.isFinite(p30.bikes)) ? p30.bikes : null,
                    alerted: ok ? (shortageAlert || fullAlert) : false,
                    shortageAlert: ok ? shortageAlert : false,
                    fullAlert: ok ? fullAlert : false,
                });
            }
        }

        const statsByKey = new Map();
        for (const [key, rec] of perKey.entries()) {
            // 依基準時段排序，確保相鄰性正確
            rec.series.sort((a, b) => a.slot.localeCompare(b.slot));
            const valid = rec.series.filter(s => s.valid);
            const validCount = valid.length;

            if (validCount < MIN_VALID_SLOTS) {
                statsByKey.set(key, {
                    available: false, validCount,
                    avgOccupancy: null, busiestTarget: null, riskHours: null,
                    name: rec.name, district: rec.district, cluster: rec.cluster, total: rec.total,
                    series: rec.series,
                });
                continue;
            }
            // 平均滿載率 = 有效時段預測可借比例平均
            const avg = valid.reduce((s, x) => s + x.bikeRatio, 0) / validCount;
            // 最忙時段 = 相鄰『有效』基準時段間預測車數變動最大者（顯示其 +30 目標時間）
            let maxDelta = -1, busiestTarget = null;
            for (let i = 1; i < rec.series.length; i++) {
                const cur = rec.series[i], prev = rec.series[i - 1];
                if (!cur.valid || !prev.valid || cur.bikes == null || prev.bikes == null) continue;
                const delta = Math.abs(cur.bikes - prev.bikes);
                if (delta > maxDelta) { maxDelta = delta; busiestTarget = cur.target; }
            }
            // 風險時數 = +30 缺車/滿站預警時段數 × 0.5 小時
            const alertedSlots = valid.filter(s => s.alerted).length;
            const riskHours = alertedSlots * 0.5;

            statsByKey.set(key, {
                available: true, validCount,
                avgOccupancy: avg,                 // 比例（0..1）
                busiestTarget: busiestTarget,      // '+30 目標時間' HH:MM 或 null（無相鄰有效對）
                busiestDelta: maxDelta >= 0 ? maxDelta : null,
                riskHours: riskHours,
                alertedSlots,
                name: rec.name, district: rec.district, cluster: rec.cluster, total: rec.total,
                series: rec.series,                // 每半小時 +30 預警序列（供連續模型預警判斷）
            });
        }
        return { statsByKey, thresholds };
    }

    // 對外：確保某回放日期的整日統計已就緒（快取）；回報進度。
    async function ensureDayStats(onProgress) {
        const date = REPLAY_DATE();
        if (_dayCache.has(date)) return _dayCache.get(date);
        if (_loadingPromise && _loadingDate === date) return _loadingPromise;
        _loadingDate = date;
        _loadingPromise = (async () => {
            const slots = slots48();
            const slotResults = await fetchAllSlots(date, slots, onProgress);
            const okCount = slotResults.filter(r => r && r.body && Array.isArray(r.body.stations)).length;
            const agg = aggregate(slotResults);
            const entry = {
                date, statsByKey: agg.statsByKey, thresholds: agg.thresholds,
                slotsFetched: slots.length, slotsOk: okCount, builtAt: Date.now(),
            };
            _dayCache.set(date, entry);
            _loadingPromise = null; _loadingDate = null;
            return entry;
        })();
        return _loadingPromise;
    }

    window.PatternDayStats = {
        ensureDayStats,
        getStats(stationKey) {
            const date = REPLAY_DATE();
            const entry = _dayCache.get(date);
            if (!entry) return null;
            return entry.statsByKey.get(stationKey) || null;
        },
        // 回傳某站整日 +30 預警序列（依基準時段排序）。缺資料回 null。
        getSeries(stationKey) {
            const entry = _dayCache.get(REPLAY_DATE());
            if (!entry) return null;
            const st = entry.statsByKey.get(stationKey);
            return (st && st.series) ? st.series : null;
        },
        isReady() { return _dayCache.has(REPLAY_DATE()); },
        currentEntry() { return _dayCache.get(REPLAY_DATE()) || null; },
        replayDate: REPLAY_DATE,
    };
}());
