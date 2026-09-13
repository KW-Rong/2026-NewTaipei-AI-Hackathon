/* 把前端版面接到『本次專案四個 SageMaker XGBoost 模型』的可追溯回放資料 (appData.js)。
 *
 * 重要原則（與後端 model_meta.json 一致）：
 * - 模型直接輸出的是『可借／可還比例』，非機率。畫面只顯示「預測比例」與「警示門檻」，
 *   不使用任何以百分比表述的機率字樣（除非另有經驗證的機率校準）。
 * - 四個模型各有 5 月校正、6 月驗證的門檻：
 *     bike-30 / dock-30 / bike-60 / dock-60。
 *   缺車預警：預測『可借比例 <= bikeThreshold』；滿站預警：預測『可還比例 <= dockThreshold』。
 * - 即時事件（目前車況）與未來預警分開：即時事件依當下實際比例 <= 0.10（事件定義）。
 * - 資料不足（該時點缺觀測或歷史特徵不足）不當 0，明確標「資料不足」。
 * - 尖峰／調度優先序／路線屬於另外的歷史摘要／規則，與模型直接預測分開，且標為預覽。
 */
(function () {
    const meta = appData.meta;
    const TH = meta.warningThresholds;                 // {bike-30,dock-30,bike-60,dock-60}
    const EVENT_RATIO = 0.10;                           // 事件定義：真值比例 <= 0.10 視為缺車/滿站
    // 四種 K-Means 供需型態（固定順序）；未分群新站不在此列，屬資料狀態。
    const MODEL_CLUSTERS = ['平穩低波動型', '均衡流動型', '雙尖峰高流動型', '通勤到達型'];

    // 由當前 stationData 動態計算各型態站數（含未分群新站）。
    function computeClusterCounts() {
        const counts = {};
        stationData.forEach(s => { counts[s.pattern] = (counts[s.pattern] || 0) + 1; });
        return counts;
    }

    const countFormat = new Intl.NumberFormat('zh-TW');
    const pct = value => (value == null || !Number.isFinite(value)) ? '—' : (value * 100).toFixed(1) + '%';
    let alertKindFilter = 'ALL';
    let overviewPersistentAlertKind = 'shortage';
    let overviewPersistentAlertHours = 1;

    function escapeHtml(value) {
        return String(value ?? '').replace(/[&<>'"]/g, char => ({
            '&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#39;', '"': '&quot;'
        }[char]));
    }

    function rowAt(station, index = currentTimelineIndex) {
        return station.timeline?.[(index + 48) % 48] || null;
    }

    // timeline row 語意：
    // [0]現在車數 [1]+30車數 [2]+60車數 [3]+30可借比例 [4]+30可還比例 [5]狀態
    // [6]+60可借比例 [7]+60可還比例 [8]+30空位數 [9]+60空位數
    // 站況資料來源有兩種：
    //  (A) 回放 timeline：station.timeline[48格]（appData.js 歷史離線備援）
    //  (B) SageMaker MME 單點預測：station.mmePred = {
    //        status:'ok'|'insufficient'|'stale', bikesNow, total,
    //        p:{30:{bikeRatio,dockRatio,bikes,docks},60:{...}} }
    // 五項主畫面功能皆透過以下存取器取值；改存取器即可讓地圖/列表/詳情/警示/調度同時支援兩者。
    function isMmeStation(station) { return !!(station && station.mmePred); }

    function slotStatusOf(station, index = currentTimelineIndex) {
        if (isMmeStation(station)) return station.mmePred.status || 'missing';
        return station.slotStatus?.[(index + 48) % 48] || 'missing';
    }
    function isMissing(row) { return !row || row[0] === null || row[0] === undefined; }

    function mmeMissing(station) {
        const m = station.mmePred;
        return !m || m.status !== 'ok' || m.bikesNow == null;
    }

    // 取指定 offset 的『車數』；資料不足回 null（不填 0）。
    function bikesAt(station, minutes) {
        if (isMmeStation(station)) {
            const m = station.mmePred;
            if (m.status !== 'ok') return (minutes === 0 && m.bikesNow != null) ? m.bikesNow : null;
            if (minutes === 0) return m.bikesNow != null ? m.bikesNow : null;
            const p = m.p && m.p[String(minutes)];
            return (p && Number.isFinite(p.bikes)) ? p.bikes : null;
        }
        const row = rowAt(station);
        if (isMissing(row)) return null;
        const value = minutes === 60 ? row[2] : minutes === 30 ? row[1] : row[0];
        return Number.isFinite(value) ? value : null;
    }
    // 取指定 horizon 的預測可借比例；horizon=0 時取當下實際比例（用後端未取整 curBikeRatio，與 KPI/事件同源）。
    function bikeRatioAt(station, horizon) {
        if (isMmeStation(station)) {
            const m = station.mmePred;
            if (horizon === 0) {
                if (m.curBikeRatio != null) return m.curBikeRatio;
                return (m.bikesNow != null && m.total) ? m.bikesNow / m.total : null;
            }
            if (m.status !== 'ok') return null;
            const p = m.p && m.p[String(horizon)];
            return (p && Number.isFinite(p.bikeRatio)) ? p.bikeRatio : null;
        }
        const row = rowAt(station);
        if (isMissing(row)) return null;
        if (horizon === 60) return row[6];
        if (horizon === 30) return row[3];
        return station.total ? row[0] / station.total : null;   // 當下可借比例
    }
    function dockRatioAt(station, horizon) {
        if (isMmeStation(station)) {
            const m = station.mmePred;
            if (horizon === 0) {
                if (m.curDockRatio != null) return m.curDockRatio;
                return (m.bikesNow != null && m.total) ? Math.max(0, m.total - m.bikesNow) / m.total : null;
            }
            if (m.status !== 'ok') return null;
            const p = m.p && m.p[String(horizon)];
            return (p && Number.isFinite(p.dockRatio)) ? p.dockRatio : null;
        }
        const row = rowAt(station);
        if (isMissing(row)) return null;
        if (horizon === 60) return row[7];
        if (horizon === 30) return row[4];
        // 當下可還比例：容量 - 現在車數 - 待修，近似 (total - bikes)/total
        return station.total ? Math.max(0, station.total - row[0]) / station.total : null;
    }

    // 依『目前選定 horizon（0/30/60）』判斷警示。
    // 回傳 { kind:'shortage'|'full', horizon, ratio, threshold, immediate } 或 null。
    function warningAt(station) {
        // 資料不足不判警示（兩種來源分別判斷）
        if (isMmeStation(station)) {
            if (mmeMissing(station)) return null;
            // MME 站：一律以『後端 authoritative alerts』為單一真相來源，前端不重算。
            // 後端 immediate 事件用未取整的當下比例判斷；前端若改用取整後 curBikeRatio 重算，
            // 會在比例正好等於門檻（0.10）時翻轉（曾造成缺車 394/381、滿站 50/45 的差異）。
            return warningFromBackendAlerts(station);
        } else if (isMissing(rowAt(station))) {
            return null;
        }

        // ── 以下為 appData.js 歷史離線備援（非 MME）：沿用 timeline 重算 ──
        // 即時事件（基準時點）：當下實際比例 <= 事件定義 0.10
        const curBikeRatio = bikeRatioAt(station, 0);
        const curDockRatio = dockRatioAt(station, 0);
        if (curBikeRatio != null && curBikeRatio <= EVENT_RATIO) {
            return { kind: 'shortage', horizon: 0, ratio: curBikeRatio, threshold: EVENT_RATIO, immediate: true };
        }
        if (curDockRatio != null && curDockRatio <= EVENT_RATIO) {
            return { kind: 'full', horizon: 0, ratio: curDockRatio, threshold: EVENT_RATIO, immediate: true };
        }
        if (currentForecastMinutes === 0) return null;

        // 未來預警：用該 horizon 的模型預測比例 vs 該向門檻
        const horizon = currentForecastMinutes === 60 ? 60 : 30;
        if (slotStatusOf(station) !== 'ok') return null;       // 歷史特徵不足→不發預警
        const bikeRatio = bikeRatioAt(station, horizon);
        const dockRatio = dockRatioAt(station, horizon);
        const bikeTh = horizon === 60 ? TH['bike-60'] : TH['bike-30'];
        const dockTh = horizon === 60 ? TH['dock-60'] : TH['dock-30'];
        if (bikeRatio != null && bikeRatio <= bikeTh) {
            return { kind: 'shortage', horizon, ratio: bikeRatio, threshold: bikeTh, immediate: false };
        }
        if (dockRatio != null && dockRatio <= dockTh) {
            return { kind: 'full', horizon, ratio: dockRatio, threshold: dockTh, immediate: false };
        }
        return null;
    }

    // 依『目前選定 horizon（0/30/60）』從後端 alerts 挑出對應警示。
    // 規則與後端 predictor.py 完全一致：immediate 事件優先於未來預警。
    // - currentForecastMinutes===0：只回目前事件（immediate）。
    // - 30/60：先回目前事件（若存在），否則回該 horizon 的未來預警。
    function warningFromBackendAlerts(station) {
        const alerts = (station.mmePred && Array.isArray(station.mmePred.alerts)) ? station.mmePred.alerts : [];
        if (!alerts.length) return null;
        const immediate = alerts.find(a => a.immediate);
        if (immediate) {
            return { kind: immediate.kind, horizon: 0, ratio: immediate.ratio,
                     threshold: immediate.threshold, immediate: true };
        }
        if (currentForecastMinutes === 0) return null;   // 目前事件檢視：無 immediate 即無警示
        const horizon = currentForecastMinutes === 60 ? 60 : 30;
        // 同 horizon 若同時有缺車與滿站，維持後端順序（缺車在前）並取第一個。
        const hit = alerts.find(a => !a.immediate && a.horizon === horizon);
        if (hit) {
            return { kind: hit.kind, horizon: hit.horizon, ratio: hit.ratio,
                     threshold: hit.threshold, immediate: false };
        }
        return null;
    }

    function filteredStations() {
        return stationData.filter(station =>
            (currentDistrict === 'ALL' || station.district === currentDistrict)
            && (currentPatternFilter === 'ALL' || station.pattern === currentPatternFilter)
        );
    }

    // 地圖實際可見站（行政區 + 供需型態 + 地圖 empty/full 篩選），與 renderMarkers 顯示一致。
    // 右側詳情面板的同步以此為準。
    function visibleStationsForPanel() {
        return filteredStations().filter(station => {
            if (currentFilter === 'empty' || currentFilter === 'full') {
                const w = warningAt(station);
                const kind = w ? w.kind : null;
                if (currentFilter === 'empty' && kind !== 'shortage') return false;
                if (currentFilter === 'full' && kind !== 'full') return false;
            }
            // 「有待修車」：目前無真實未來待修車預測資料，故此篩選無可顯示站點（不冒充）。
            if (currentFilter === 'fault') return false;
            return true;
        });
    }

    // 清空右側站點詳情（篩選後無站點時）。不留上一站數字。
    function clearStationPanel(message) {
        currentSelectedStationId = null;
        const setTxt = (id, v) => { const el = document.getElementById(id); if (el) el.textContent = v; };
        setTxt('panel-station-name', message || '此篩選無站點');
        setTxt('panel-station-sub', '請調整行政區、供需型態或地圖篩選');
        setTxt('panel-station-type-tag', '—');
        ['panel-bikes-now', 'panel-bikes-30', 'panel-bikes-60'].forEach(id => setTxt(id, '—'));
        const exp = document.getElementById('panel-forecast-explanation');
        if (exp) exp.innerHTML = '<div class="text-slate-500">目前篩選條件下沒有可顯示的站點。</div>';
        const title = document.getElementById('panel-recommendation-title');
        if (title) title.textContent = '建議介入：—';
        const rec = document.getElementById('gov-decision-recommendation');
        if (rec) rec.textContent = '此篩選無站點，暫無建議。';
        if (typeof chartForecastBikes !== 'undefined' && chartForecastBikes) {
            try { chartForecastBikes.destroy(); } catch (e) {}
            chartForecastBikes = null;
        }
    }

    // 切換行政區／供需型態／地圖篩選後，讓右側詳情與可見站同步：
    //  - 目前所選站仍在可見集合 → 保留並重繪
    //  - 否則改選可見集合第一站
    //  - 可見集合為空 → 清空並顯示「此篩選無站點」
    window.syncPanelToVisible = function () {
        const visible = visibleStationsForPanel();
        if (!visible.length) { clearStationPanel('此篩選無站點'); return; }
        const stillVisible = currentSelectedStationId
            && visible.some(s => s.id === currentSelectedStationId);
        if (stillVisible) {
            const s = visible.find(x => x.id === currentSelectedStationId);
            if (typeof selectStationForPanel === 'function') selectStationForPanel(s.id);
        } else {
            if (typeof selectStationForPanel === 'function') selectStationForPanel(visible[0].id);
        }
    };

    function currentAlerts(includeImmediate = true) {
        return filteredStations().map(station => ({ station, warning: warningAt(station) }))
            .filter(item => item.warning && (includeImmediate || !item.warning.immediate))
            .sort((a, b) => {
                const priority = warning => warning.kind === 'full'
                    ? (warning.immediate ? 0 : 1)
                    : (warning.immediate ? 2 : 3);
                const priorityDifference = priority(a.warning) - priority(b.warning);
                if (priorityDifference) return priorityDifference;
                // 越接近門檻（比例越低）越優先
                return a.warning.ratio - b.warning.ratio;
            });
    }

    function hydrateMetadata() {
        // override（MME 模式）：{ dataPeriod, dataTime, sourceNote, modelVersionText, modeText,
        //   observationText, timelineDateText }。未帶入則沿用 appData.meta（離線備援）。
        const ov = (typeof arguments[0] === 'object' && arguments[0]) ? arguments[0] : null;
        // 站數/行政區數一律由『當前 stationData』動態計算（MME=1562；離線備援=appData 站數）。
        const totalStations = stationData.length;
        const districtSet = new Set(stationData.map(s => s.district));
        document.getElementById('sidebar-station-count').textContent = `${countFormat.format(totalStations)}站`;
        document.getElementById('sidebar-observation-count').textContent =
            ov ? (ov.observationText || '—')
               : `${(meta.observationCount / 10000).toFixed(1)}萬筆回放`;
        document.getElementById('header-data-period').textContent = ov
            ? `${ov.dataPeriod}（${ov.dataTime}｜${countFormat.format(totalStations)} 站）`
            : `${meta.dataPeriod}（回放日 ${meta.replayDate}｜${countFormat.format(totalStations)} 站）`;
        document.getElementById('kpi-station-count').innerHTML =
            `${countFormat.format(totalStations)} <span class="text-xs font-normal text-slate-500">站</span>`;
        document.getElementById('kpi-district-count').textContent = `資料涵蓋 ${districtSet.size} 個行政區`;
        document.getElementById('overview-timeline-date').textContent = ov
            ? (ov.timelineDateText || ov.dataTime)
            : `${meta.replayDate.replaceAll('-', '/')} 模型回放日（歷史）`;

        // 頁首單行來源（取代已移除的資料模式橫幅）：MME 模式顯示端點來源；離線備援顯示靜態。
        const headerSrcEl = document.getElementById('header-model-source');
        if (headerSrcEl) headerSrcEl.textContent = ov
            ? (ov.headerSource || 'SageMaker 多模型 Endpoint 真實推論')
            : 'appData.js 靜態離線備援（非線上 MME）';

        // 來源模式／基準時間／資料更新／模型版本（監理稽核用）
        const modeEl = document.getElementById('sidebar-mode');
        if (modeEl) modeEl.textContent = ov ? (ov.modeText || 'SageMaker MME') : `${meta.replayDate} 歷史回放`;
        const freshEl = document.getElementById('sidebar-model-version');
        if (freshEl) freshEl.textContent = ov ? (ov.modelVersionText || '—') : `${meta.modelVersion.runId}（-r2）`;
        const noteEl = document.getElementById('data-source-note');
        if (noteEl) noteEl.textContent = ov ? (ov.sourceNote || '') : meta.sourceNote;
    }

    function populateFilters() {
        const patternSelect = document.getElementById('header-pattern-select');
        // 四群統計、未分群數皆由當前 stationData 動態計算（不依賴 meta 預算值）。
        // 未分群新站是「資料狀態」，不是第五種 K-Means 型態：固定排在四型之後、樣式區隔。
        const counts = computeClusterCounts();
        const unclusteredLabel = meta.unclusteredLabel || '未分群新站';
        const modelClusters = MODEL_CLUSTERS.filter(c => counts[c]);
        const modelCount = modelClusters.length;
        const unclusteredCount = counts[unclusteredLabel] || 0;
        const totalStations = stationData.length;
        const allLabel = unclusteredCount
            ? `全部 ${modelCount} 種供需型態＋${countFormat.format(unclusteredCount)} 個${unclusteredLabel}（共 ${countFormat.format(totalStations)} 站）`
            : `全部 ${modelCount} 種供需型態（${countFormat.format(totalStations)} 站）`;
        patternSelect.replaceChildren(new Option(allLabel, 'ALL'));
        // 先四型，再未分群新站（明確區隔，不併入四群）
        modelClusters.forEach(name => patternSelect.add(new Option(`${name}（${countFormat.format(counts[name])} 站）`, name)));
        if (unclusteredCount) {
            patternSelect.add(new Option(`${unclusteredLabel}（${countFormat.format(unclusteredCount)} 站，資料狀態）`, unclusteredLabel));
        }
        patternSelect.value = 'ALL';

        const districtSelect = document.getElementById('district-filter');
        const districtCounts = Object.entries(stationData.reduce((result, station) => {
            result[station.district] = (result[station.district] || 0) + 1;
            return result;
        }, {})).sort((a, b) => a[0].localeCompare(b[0], 'zh-Hant'));
        // 行政區數由當前 stationData 動態計算（不用 meta.districtCount 靜態值）。
        const districtCount = districtCounts.length;
        districtSelect.replaceChildren(new Option(`全資料集（${districtCount} 行政區）`, 'ALL'));
        districtCounts.forEach(([district, count]) => districtSelect.add(new Option(`${district}（${count} 站）`, district)));
        districtSelect.value = 'ALL';
        // 全區站數用當前 stationData（MME=1562），不用 meta.stationCount（appData 靜態 1576）。
        document.getElementById('current-pattern-badge').textContent = `全區 ${countFormat.format(totalStations)} 站`;
    }

    window.stationBikesForOffset = function (station, offsetMinutes) {
        return bikesAt(station, offsetMinutes);   // 可能為 null（資料不足）
    };

    window.stationBikesAtSelectedTime = function (station) {
        return bikesAt(station, currentForecastMinutes);
    };

    // 供站點分類表使用：回傳該站『真正由 API 預測產生』的 +30／+60 可借/可還比例與預警。
    // 全部來自 MME 模型輸出（bike/dock × 30/60）；資料不足回 null（呼叫端顯示「資料不足」，不補造）。
    // warning 直接沿用 warningAt（MME 站讀後端 authoritative alerts）。
    window.stationPredictionSummary = function (station) {
        const missing = isMmeStation(station) ? mmeMissing(station) : isMissing(rowAt(station));
        if (missing) {
            return { available: false, b30: null, d30: null, b60: null, d60: null, warning: null };
        }
        return {
            available: true,
            b30: bikeRatioAt(station, 30), d30: dockRatioAt(station, 30),
            b60: bikeRatioAt(station, 60), d60: dockRatioAt(station, 60),
            warning: warningAt(station),
        };
    };
    // 百分比格式化（與畫面一致）；null → 資料不足。
    window.formatRatioPct = function (v) {
        return (v == null || !Number.isFinite(v)) ? '資料不足' : (v * 100).toFixed(1) + '%';
    };

    window.updateOverviewRiskCounts = function () {
        // KPI 一律計『未來預警』（排除 immediate 目前事件）；horizon 依時間軸滑桿選擇。
        const hz = currentForecastMinutes === 60 ? 60 : (currentForecastMinutes === 30 ? 30 : 30);
        const alerts = currentAlerts(false);
        const shortage = alerts.filter(item => item.warning.kind === 'shortage').length;
        const full = alerts.filter(item => item.warning.kind === 'full').length;
        document.getElementById('kpi-empty-count').innerHTML = `${shortage} <span class="text-xs font-normal text-slate-500">站</span>`;
        document.getElementById('kpi-full-count').innerHTML = `${full} <span class="text-xs font-normal text-slate-500">站</span>`;
        const subS = document.getElementById('kpi-empty-sub');
        const subF = document.getElementById('kpi-full-sub');
        const label = currentForecastMinutes === 0
            ? '請選 +30/+60 檢視未來預警（非目前事件）'
            : `+${currentForecastMinutes} 分鐘模型預警（比例≤門檻，非目前事件）`;
        if (subS) subS.textContent = label;
        if (subF) subF.textContent = label;
        renderAlerts();
        renderRelayCandidates();
        buildDispatchCandidate();
        renderOverviewPersistentAlerts();
    };

    window.renderMarkers = function () {
        if (!markersGroup) return;
        markersGroup.clearLayers();
        filteredStations().forEach(station => {
            const bikes = bikesAt(station, currentForecastMinutes);
            const missing = bikes == null;
            const warning = warningAt(station);
            let kind = warning ? warning.kind : null;
            if (currentFilter === 'empty' && kind !== 'shortage') return;
            if (currentFilter === 'full' && kind !== 'full') return;
            // 「有待修車」：目前『沒有』真實的未來待修車預測資料（四個 MME 模型不含維修目標，
            // 且無逐站逐時的維修標籤）。故此篩選不顯示任何站，也不得用 ratio_unavailable/停用/
            // 缺車/滿站冒充維修。畫面於下方以說明列標示「尚無待修車預測資料」。
            if (currentFilter === 'fault') return;

            const color = missing ? '#94a3b8' : kind === 'shortage' ? '#ef4444' : kind === 'full' ? '#f59e0b' : '#10b981';
            const animation = kind === 'shortage' ? 'marker-alert-empty' : kind === 'full' ? 'marker-alert-full' : '';
            const label = missing ? '—' : bikes;
            const icon = L.divIcon({
                className: 'custom-leaflet-marker',
                html: `<div class="relative flex items-center justify-center w-8 h-8 rounded-full bg-white font-bold text-xs shadow-md ${animation}" style="border:2.5px solid ${color}"><span style="color:${color}">${label}</span></div>`,
                iconSize: [32, 32], iconAnchor: [16, 16]
            });
            const marker = L.marker([station.lat, station.lng], { icon });
            const forecastLine = missing
                ? `<div class="text-xs bg-slate-50 p-1.5 rounded border mt-1">此時段：<strong>資料不足</strong></div>`
                : `<div class="text-xs bg-slate-50 p-1.5 rounded border mt-1">${currentForecastMinutes ? `+${currentForecastMinutes} 分鐘預測` : '基準時點'}：<strong>${bikes} / ${station.total}</strong></div>`;
            const warnLine = (warning && !warning.immediate)
                ? `<div class="text-[10px] mt-1 text-slate-600">預測${warning.kind === 'shortage' ? '可借' : '可還'}比例 ${pct(warning.ratio)}（門檻 ${pct(warning.threshold)}）</div>`
                : (warning && warning.immediate)
                    ? `<div class="text-[10px] mt-1 text-red-600">即時事件：目前${warning.kind === 'shortage' ? '可借' : '可還'}比例 ${pct(warning.ratio)}</div>`
                    : '';
            marker.bindPopup(`<div class="p-1 font-sans text-slate-800"><div class="font-bold text-xs">${escapeHtml(station.name)}</div><div class="text-[10px] text-blue-600 font-bold">${escapeHtml(station.pattern)}</div>${forecastLine}${warnLine}<button onclick="selectStationForPanel('${station.id}')" class="mt-2 w-full py-1 bg-blue-600 text-white rounded text-[10px] font-bold">查看詳細時段</button></div>`);
            marker.on('click', () => selectStationForPanel(station.id));
            markersGroup.addLayer(marker);
        });
        updateOverviewRiskCounts();
    };

    window.renderStationForecast = function (station) {
        const values = [0, 30, 60].map(minutes => bikesAt(station, minutes));
        const showVal = v => (v == null ? '資料不足' : v);
        document.getElementById('panel-bikes-now').textContent = showVal(values[0]);
        document.getElementById('panel-bikes-30').textContent = showVal(values[1]);
        document.getElementById('panel-bikes-60').textContent = showVal(values[2]);

        const warning = warningAt(station);
        const status = slotStatusOf(station);
        const b30 = bikeRatioAt(station, 30), d30 = dockRatioAt(station, 30);
        const b60 = bikeRatioAt(station, 60), d60 = dockRatioAt(station, 60);

        const stationMissing = isMmeStation(station) ? mmeMissing(station) : isMissing(rowAt(station));
        let predictionText;
        if (stationMissing) {
            predictionText = '此站在該時段沒有可追溯的觀測與預測，顯示為資料不足。';
        } else if (status !== 'ok') {
            predictionText = '此站在該時點歷史特徵不足（未滿過去 2 小時連續觀測），僅顯示當下車況，不產生預警。';
        } else {
            predictionText =
                `+30 分：預測可借比例 ${pct(b30)}（缺車門檻 ${pct(TH['bike-30'])}）、可還比例 ${pct(d30)}（滿站門檻 ${pct(TH['dock-30'])}）。`
                + `　+60 分：可借 ${pct(b60)}（門檻 ${pct(TH['bike-60'])}）、可還 ${pct(d60)}（門檻 ${pct(TH['dock-60'])}）。`;
        }
        const geoText = station.geoTags?.length
            ? `此站 300 公尺周邊標籤：${escapeHtml(station.geoTags.join('、'))}（僅為場站分類特徵，非因果結論）。`
            : '此站無周邊分類標籤。';
        document.getElementById('panel-forecast-explanation').innerHTML =
            `<div><span class="font-bold text-amber-700">模型預測（比例）：</span>${predictionText}</div>`
            + `<div class="mt-1 text-slate-500">${geoText}</div>`
            + `<div class="mt-1 text-slate-500">四個模型 6 月測試 MAE（比例）：bike-30 ${meta.testMetrics['bike-30'].MAE_ratio}／bike-60 ${meta.testMetrics['bike-60'].MAE_ratio}。此為比例回歸模型，畫面數值為預測比例，非機率。</div>`;

        const title = document.getElementById('panel-recommendation-title');
        const recommendation = document.getElementById('gov-decision-recommendation');
        if (warning?.kind === 'shortage') {
            title.textContent = warning.immediate ? '即時事件：人工確認補車' : `${warning.horizon} 分鐘預警：人工確認補車`;
            recommendation.textContent = `依門檻規則：預測可借比例 ${pct(warning.ratio)} ≤ 門檻 ${pct(warning.threshold)}。請先核對即時站況與鄰近可抽車站後再建立工單（不會自動派車）。`;
        } else if (warning?.kind === 'full') {
            title.textContent = warning.immediate ? '即時事件：人工確認移車' : `${warning.horizon} 分鐘預警：人工確認移車`;
            recommendation.textContent = `依門檻規則：預測可還比例 ${pct(warning.ratio)} ≤ 門檻 ${pct(warning.threshold)}。請先核對即時站況與鄰近可容納站後再建立工單（不會自動派車）。`;
        } else {
            title.textContent = '建議介入：持續監測';
            recommendation.textContent = stationMissing
                ? '此時段資料不足，暫不評估。'
                : '目前未跨入任何警示門檻。';
        }
        document.getElementById('panel-station-sub').textContent =
            `${station.district} · ${station.total} 個車柱 · ${station.pattern}${station.geoTags?.length ? ' · ' + station.geoTags.join('、') : ''}`;

        const canvas = document.getElementById('chart-forecast-bikes');
        if (!canvas) return;
        if (chartForecastBikes) chartForecastBikes.destroy();
        const chartValues = values.map(v => (v == null ? null : v));
        chartForecastBikes = new Chart(canvas.getContext('2d'), {
            type: 'line', data: { labels: ['基準', '+30 分', '+60 分'], datasets: [{ data: chartValues, borderColor: '#2563eb', backgroundColor: 'rgba(37,99,235,.12)', borderWidth: 3, fill: true, tension: .2, pointRadius: 5, spanGaps: false }] },
            options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false } }, scales: { y: { beginAtZero: true, max: Math.max(station.total, ...chartValues.filter(Number.isFinite), 1), ticks: { precision: 0 } } } }
        });
    };

    window.handleDistrictChange = function (district) {
        currentDistrict = district;
        const visible = district === 'ALL' ? stationData : stationData.filter(station => station.district === district);
        if (visible.length && map) {
            const bounds = L.latLngBounds(visible.map(station => [station.lat, station.lng]));
            map.fitBounds(bounds, { padding: [25, 25], maxZoom: district === 'ALL' ? 12 : 14 });
        }
        renderMarkers();
        window.syncPanelToVisible();   // 右側詳情與新篩選同步（保留/改選第一站/清空）
        renderAlerts();
        renderRelayCandidates();
        buildDispatchCandidate();
    };

    window.filterByKMeansPattern = function (pattern) {
        if (activeFilterPage === 'patterns') {
            tablePatternFilter = pattern; handleStationSearch(patternSearchQuery);
        } else if (activeFilterPage === 'gis') {
            gisPatternFilter = pattern; renderGisPatterns();
        } else {
            currentPatternFilter = pattern;
            const counts = computeClusterCounts();
            const count = pattern === 'ALL' ? stationData.length : (counts[pattern] || 0);
            document.getElementById('current-pattern-badge').textContent = pattern === 'ALL' ? `全區 ${count} 站` : `${pattern} · ${count} 站`;
            renderMarkers();
            window.syncPanelToVisible();   // 右側詳情與新型態篩選同步
        }
    };

    function levelText(warning) {
        const kindText = warning.kind === 'shortage' ? '缺車' : '滿站';
        return warning.immediate ? `P1-即時${kindText}` : `P3-${kindText}預警`;
    }

    function renderAlerts() {
        const matchingAlerts = currentAlerts().filter(({ warning }) =>
            alertKindFilter === 'ALL' || warning.kind === alertKindFilter
        );
        const alerts = matchingAlerts.slice(0, 100);
        const tbody = document.getElementById('alert-station-tbody');
        tbody.innerHTML = alerts.length ? alerts.map(({ station, warning }) => {
            const bikes = bikesAt(station, 0);
            const ratioText = bikes == null ? '資料不足' : `${bikes} / ${station.total} (${(bikes / station.total * 100).toFixed(1)}%)`;
            const shortage = warning.kind === 'shortage';
            const colors = shortage
                ? 'bg-red-100 text-red-700 border border-red-200'
                : 'bg-amber-100 text-amber-700 border border-amber-200';
            const detail = warning.immediate
                ? `基準時點事件 · 比例 ${pct(warning.ratio)}`
                : `+${warning.horizon} 分 · 預測比例 ${pct(warning.ratio)}（門檻 ${pct(warning.threshold)}）`;
            return `<tr><td class="p-3"><span class="px-2 py-0.5 ${colors} rounded font-bold">${levelText(warning)}</span></td><td class="p-3 font-sans font-bold text-slate-800">${escapeHtml(station.name)}</td><td class="p-3 font-sans">${escapeHtml(station.pattern)}</td><td class="p-3 font-bold">${ratioText}</td><td class="p-3 font-bold">${detail}</td><td class="p-3 font-sans"><button onclick="switchTab('trucks')" class="px-2.5 py-1 bg-blue-600 text-white font-bold rounded-lg">查看調度候選</button></td></tr>`;
        }).join('') : '<tr><td colspan="6" class="p-6 text-center text-slate-400">此篩選條件與時段沒有警示</td></tr>';
        const total = matchingAlerts.length;
        ['sidebar-alert-count', 'header-alert-badge'].forEach(id => document.getElementById(id).textContent = total > 999 ? '999+' : total);
        const filterText = alertKindFilter === 'shortage' ? '缺車' : alertKindFilter === 'full' ? '滿站' : '全部';
        document.getElementById('alert-sort-status').textContent =
            `${filterText}：顯示前 ${Math.min(100, total)} / ${total} 筆 · 門檻（可借/可還比例）缺車30 ${pct(TH['bike-30'])}、滿站30 ${pct(TH['dock-30'])}、缺車60 ${pct(TH['bike-60'])}、滿站60 ${pct(TH['dock-60'])}`;
    }
    window.setAlertKindFilter = function (kind) {
        alertKindFilter = ['shortage', 'full'].includes(kind) ? kind : 'ALL';
        renderAlerts();
    };

    function renderRelayCandidates() {
        const candidates = currentAlerts(false).slice(0, 5);
        const list = document.getElementById('relay-risk-list');
        list.innerHTML = candidates.length ? candidates.map(({ station, warning }) => {
            const color = warning.kind === 'shortage' ? 'red' : 'amber';
            const statusText = warning.kind === 'shortage' ? '缺車風險' : '滿站風險';
            return `<button type="button" onclick="selectSmsTarget('${escapeHtml(station.district)}','${escapeHtml(station.name)}','${statusText}')" class="block w-full text-left p-3 rounded-xl border border-${color}-200 bg-${color}-50/50 hover:bg-${color}-100 transition"><div class="flex justify-between font-bold text-${color}-700"><span>${escapeHtml(station.name)}</span><span>${pct(warning.ratio)}</span></div><p class="text-[11px] text-slate-600 mt-1">+${warning.horizon} 分 ${statusText}候選（預測比例，非機率）；僅供預覽。</p></button>`;
        }).join('') : '<p class="p-4 text-slate-400 border rounded-xl">目前沒有模型候選</p>';
    }

    window.selectSmsTarget = function (district, stationName, status) {
        document.getElementById('sms-target-station').value = `${stationName}（${district}）`;
        document.getElementById('sms-content-preview').textContent =
            `[內容預覽] ${stationName} 出現${status}，可提示使用者查看鄰近站點。實際文案、受眾與獎勵需經主管機關核定。`;
        document.getElementById('sms-audience').value = '缺會員定位、同意狀態與簡訊 API，無法計算';
    };

    window.triggerSmsDispatch = function () {
        showNotification('目前僅能預覽：尚未串會員、同意管理、優惠券與簡訊 API。');
    };

    function distanceKm(a, b) {
        const rad = value => value * Math.PI / 180;
        const dLat = rad(b.lat - a.lat), dLng = rad(b.lng - a.lng);
        const value = Math.sin(dLat / 2) ** 2 + Math.cos(rad(a.lat)) * Math.cos(rad(b.lat)) * Math.sin(dLng / 2) ** 2;
        return 6371 * 2 * Math.atan2(Math.sqrt(value), Math.sqrt(1 - value));
    }

    // 以下『當日重複／持續時間／流動強度』屬於歷史摘要規則（非模型直接輸出），用於調度優先序。
    function observedKindAt(station, index) {
        const row = rowAt(station, index);
        if (isMissing(row) || !station.total) return null;
        const bikeRatio = row[0] / station.total;
        const dockRatio = Math.max(0, station.total - row[0]) / station.total;
        if (bikeRatio <= EVENT_RATIO) return 'shortage';
        if (dockRatio <= EVENT_RATIO) return 'full';
        return null;
    }

    function dispatchSignals(station, warning) {
        // MME 單點預測模式：沒有 48 格 timeline，無法計算「當日重複／持續時間／流動強度」歷史摘要。
        // 改以『模型預測嚴重度』排序：預測比例越低（越接近/低於門檻）優先度越高。
        if (isMmeStation(station)) {
            const ratio = (warning && Number.isFinite(warning.ratio)) ? warning.ratio : 0.5;
            const severity = Math.max(0, Math.min(1, 1 - ratio / 0.30));  // 比例越低越高
            const transitScore = (station.geoTags || []).some(t => t === '捷運站' || t === '火車站') ? 5 : 0;
            const fullPriority = warning && warning.kind === 'full' ? 5 : 0;
            return {
                score: severity * 90 + transitScore + fullPriority,
                episodes: 0, durationMinutes: 0, activityRatio: 0,
                humanEscalation: false,
                reason: `模型預測${warning && warning.kind === 'full' ? '可還' : '可借'}比例 ${(ratio * 100).toFixed(1)}%（依 SageMaker MME 預測嚴重度排序；無歷史摘要）`
            };
        }
        let consecutiveSlots = 0;
        for (let index = currentTimelineIndex; index >= 0; index -= 1) {
            if (observedKindAt(station, index) !== warning.kind) break;
            consecutiveSlots += 1;
        }
        let episodes = 0;
        let previous = false;
        let totalChange = 0;
        let changes = 0;
        for (let index = 0; index < 48; index += 1) {
            const active = observedKindAt(station, index) === warning.kind;
            if (active && !previous) episodes += 1;
            previous = active;
            const row = rowAt(station, index);
            const next = rowAt(station, index + 1);
            if (!isMissing(row) && !isMissing(next)) { totalChange += Math.abs(next[0] - row[0]); changes += 1; }
        }
        const durationMinutes = consecutiveSlots * 30;
        const activityRatio = changes ? totalChange / changes / Math.max(1, station.total) : 0;
        const repeatScore = Math.min(1, episodes / 4) * 40;
        const durationScore = Math.min(1, durationMinutes / 120) * 30;
        const activityScore = Math.min(1, activityRatio / 0.12) * 20;
        const transitScore = (station.geoTags || []).some(tag => tag === '捷運站' || tag === '火車站') ? 5 : 0;
        const fullPriority = warning.kind === 'full' ? 5 : 0;
        return {
            score: repeatScore + durationScore + activityScore + transitScore + fullPriority,
            episodes, durationMinutes, activityRatio,
            humanEscalation: durationMinutes >= 120,
            reason: `當日重複 ${episodes} 段 · 已持續 ${durationMinutes} 分 · 流動強度 ${(activityRatio * 100).toFixed(1)}%${transitScore ? ' · 交通節點' : ''}（歷史摘要規則）`
        };
    }

    function compareDispatchPriority(a, b) {
        const severityTier = item => item.signals.durationMinutes >= 180
            ? 2
            : item.signals.durationMinutes >= 120 ? 1 : 0;
        return severityTier(b) - severityTier(a)
            || b.signals.score - a.signals.score
            || b.signals.durationMinutes - a.signals.durationMinutes
            || (a.warning?.ratio ?? 1) - (b.warning?.ratio ?? 1)
            || a.station.name.localeCompare(b.station.name, 'zh-Hant');
    }

    function renderOverviewPersistentAlerts() {
        const section = document.getElementById('overview-persistent-section');
        const list = document.getElementById('overview-persistent-alert-list');
        const summary = document.getElementById('overview-persistent-alert-summary');
        const districtLabel = document.getElementById('overview-alert-district');
        if (!section || section.hasAttribute('hidden') || !list || !summary || !districtLabel) return;

        const kindText = overviewPersistentAlertKind === 'shortage' ? '缺車' : '滿站';
        const districtText = currentDistrict === 'ALL' ? '全資料集' : currentDistrict;
        districtLabel.textContent = districtText;

        // 資料來源：既有 /api/replay 各半小時 SageMaker MME 的 +30 預警（PatternDayStats，共用快取）。
        if (!window.PatternDayStats) {
            summary.textContent = '連續模型預警模組尚未載入。';
            list.replaceChildren();
            return;
        }
        if (!window.PatternDayStats.isReady()) {
            // 觸發整日彙整（限並行、快取），載入中顯示進度；完成後重繪。
            summary.textContent = `${districtText}｜連續${kindText}模型預警：正在向 SageMaker MME 取得整日各半小時 +30 預警…`;
            list.replaceChildren();
            if (!renderOverviewPersistentAlerts._loading) {
                renderOverviewPersistentAlerts._loading = true;
                window.PatternDayStats.ensureDayStats((done, total) => {
                    summary.textContent = `${districtText}｜連續${kindText}模型預警：整日 +30 預警載入中（${done}/${total} 時段）…`;
                }).then(() => {
                    renderOverviewPersistentAlerts._loading = false;
                    renderOverviewPersistentAlerts();
                }).catch(e => {
                    renderOverviewPersistentAlerts._loading = false;
                    // API 失敗：不退回 appData 舊卡片，明確顯示失敗。
                    summary.textContent = `連續模型預警載入失敗：${(e && e.message) || e}（不顯示 appData 靜態卡片）。`;
                    list.replaceChildren();
                });
            }
            return;
        }

        // 基準時段（overview 時間軸目前選定）與往前連續時段數。
        const slots = (typeof overviewTimelineSlots !== 'undefined') ? overviewTimelineSlots : [];
        const baseSlot = slots[currentTimelineIndex] || null;
        const needSlots = overviewPersistentAlertHours * 2;   // 每小時 2 格（30 分）
        const kindFlag = overviewPersistentAlertKind === 'shortage' ? 'shortageAlert' : 'fullAlert';

        // 對每個（篩選後）站，檢查基準時段往前連續 needSlots 格是否皆有該類 +30 模型預警。
        let insufficientCount = 0;
        const candidates = [];
        filteredStations().forEach(station => {
            const series = window.PatternDayStats.getSeries(station.key);
            if (!series || !series.length || !baseSlot) { insufficientCount += 1; return; }
            // 以基準時段為結尾，往前取 needSlots 格（索引以 series 的 slot 對齊）
            const idx = series.findIndex(s => s.slot === baseSlot);
            if (idx < 0) { insufficientCount += 1; return; }
            const startIdx = idx - (needSlots - 1);
            if (startIdx < 0) { insufficientCount += 1; return; }   // 往前時段不足 → 資料不足
            // 需求視窗內任何一格資料不足（valid=false）→ 該站資料不足，不算 0、不補值
            const windowSlots = series.slice(startIdx, idx + 1);
            if (windowSlots.some(s => !s.valid)) { insufficientCount += 1; return; }
            // 連續：視窗內每格都要有該類預警
            const allWarned = windowSlots.every(s => s[kindFlag]);
            if (!allWarned) return;   // 非連續 → 不是候選（但資料完整，不算資料不足）
            // 計算實際往前連續長度（可能超過視窗），用於卡片顯示與排序
            let run = 0;
            for (let i = idx; i >= 0; i--) {
                if (!series[i].valid || !series[i][kindFlag]) break;
                run += 1;
            }
            const startTime = series[Math.max(0, idx - run + 1)].slot;
            candidates.push({ station, run, hours: run * 0.5, startTime });
        });

        // 排序：連續時數長者優先，再依站名。
        candidates.sort((a, b) => (b.run - a.run) || a.station.name.localeCompare(b.station.name, 'zh-Hant'));

        summary.textContent =
            `${districtText}｜基準 ${baseSlot || '—'}：往前連續${kindText}模型預警滿 ${overviewPersistentAlertHours} 小時共 ${candidates.length} 站`
            + `（連續模型預警，非實際${kindText}；資料不足 ${insufficientCount} 站不計入）`;
        list.replaceChildren();

        if (!candidates.length) {
            const empty = document.createElement('div');
            empty.className = 'md:col-span-2 xl:col-span-3 rounded-xl border border-dashed border-slate-300 bg-slate-50 p-5 text-center text-xs text-slate-500';
            empty.textContent = `基準 ${baseSlot || '—'} 往前連續${kindText}模型預警未滿 ${overviewPersistentAlertHours} 小時的站點：0 站`
                + `（此為模型預警，非實際${kindText}；另有 ${insufficientCount} 站因時段不足標為資料不足）。`;
            list.appendChild(empty);
            return;
        }

        candidates.forEach(({ station, run, hours, startTime }) => {
            const isShort = overviewPersistentAlertKind === 'shortage';
            const card = document.createElement('button');
            card.type = 'button';
            card.className = isShort
                ? 'text-left rounded-xl border border-red-200 bg-red-50/60 p-4 hover:bg-red-100 transition'
                : 'text-left rounded-xl border border-amber-200 bg-amber-50/60 p-4 hover:bg-amber-100 transition';
            card.addEventListener('click', () => {
                selectStationForPanel(station.id);
                if (map) map.setView([station.lat, station.lng], 16);
            });
            const top = document.createElement('div');
            top.className = 'flex items-start justify-between gap-3';
            const name = document.createElement('div');
            name.className = 'font-black text-sm text-slate-800';
            name.textContent = station.name;
            const duration = document.createElement('span');
            duration.className = isShort
                ? 'shrink-0 rounded-full bg-red-600 px-2.5 py-1 text-[10px] font-black text-white'
                : 'shrink-0 rounded-full bg-amber-500 px-2.5 py-1 text-[10px] font-black text-white';
            duration.textContent = `連續 ${hours} 小時`;
            top.append(name, duration);
            const detail = document.createElement('div');
            detail.className = 'mt-2 text-[11px] leading-relaxed text-slate-600';
            detail.textContent = `${station.district} · 自 ${startTime} 起連續 ${run} 個半小時皆有 +30 ${kindText}模型預警（歷史回放，非實際${kindText}）`;
            card.append(top, detail);
            list.appendChild(card);
        });
    }

    window.setOverviewPersistentAlertKind = function (kind) {
        overviewPersistentAlertKind = kind === 'full' ? 'full' : 'shortage';
        renderOverviewPersistentAlerts();
    };

    window.setOverviewPersistentAlertHours = function (hours) {
        overviewPersistentAlertHours = [1, 2, 3].includes(Number(hours)) ? Number(hours) : 1;
        renderOverviewPersistentAlerts();
    };

    function buildDispatchCandidate() {
        const ranked = currentAlerts().map(item => ({ ...item, signals: dispatchSignals(item.station, item.warning) }))
            .sort(compareDispatchPriority);
        dispatchRoutePlan.stops.splice(0);
        dispatchRoutePlan.bCandidates = [];
        dispatchRoutePlan.desiredStopCount = 4;
        if (!ranked.length) {
            dispatchRoutePlan.reason = '目前篩選區域沒有缺車或滿站警示，暫不派遣';
            const status = document.getElementById('auto-dispatch-status');
            if (status) status.textContent = dispatchRoutePlan.reason;
            renderDispatchRoutePlan();
            return;
        }
        const primary = ranked[0];
        const nearby = ranked.filter(item => item.station.id !== primary.station.id && distanceKm(primary.station, item.station) <= 1)
            .sort((a, b) => {
                const oppositeA = a.warning.kind !== primary.warning.kind ? 0 : 1;
                const oppositeB = b.warning.kind !== primary.warning.kind ? 0 : 1;
                return oppositeA - oppositeB || b.signals.score - a.signals.score || distanceKm(primary.station, a.station) - distanceKm(primary.station, b.station);
            })
            .slice(0, 3);
        const selected = [primary, ...nearby];
        selected.forEach((item, index) => dispatchRoutePlan.stops.push({
            label: String.fromCharCode(65 + index),
            stationId: item.station.id,
            name: item.station.name,
            district: item.station.district,
            query: `${item.station.lat},${item.station.lng}`,
            kind: item.warning.kind,
            task: `${item.warning.kind === 'full' ? '滿站' : '缺車'}${item.warning.immediate ? '即時事件' : '預警候選'} · ${item.signals.humanEscalation ? '等待人工介入' : '調度候選（預覽）'}`,
            priorityScore: item.signals.score,
            priorityReason: item.signals.reason,
            humanEscalation: item.signals.humanEscalation
        }));
        dispatchRoutePlan.selectedAId = primary.station.id;
        dispatchRoutePlan.reason = nearby.length ? `已列出 ${selected.length} 站調度候選（預覽）` : 'A 站 1 公里內沒有其他警示站';
        const escalations = selected.filter(item => item.signals.humanEscalation);
        const status = document.getElementById('auto-dispatch-status');
        if (status) {
            status.className = escalations.length
                ? 'rounded-xl border border-red-200 bg-red-50 p-3 text-xs font-bold text-red-700'
                : 'rounded-xl border border-blue-200 bg-blue-50 p-3 text-xs text-blue-800';
            status.textContent = escalations.length
                ? `${escalations.length} 站已持續失衡至少 2 小時：列為人工介入候選。尚未實際派車或最佳化路線（預覽）。`
                : `已依歷史摘要優先分數列出 ${selected.length} 站調度候選（預覽）；尚未實際派車。`;
        }
        renderDispatchRoutePlan();
    }
    window.rebuildDispatchCandidate = buildDispatchCandidate;

    window.confirmDispatch = function () {
        closeModal('dispatch-modal');
        showNotification('已完成前端確認示範；尚未串工單資料庫，因此不會真的派車。');
    };

    window.exportDailyReport = function () {
        const rows = [['回放日期', '基準時間', '行政區', '站點', '供需型態', '目前車數', '容量', '警示類型', '預測比例', '警示門檻']];
        currentAlerts().forEach(({ station, warning }) => {
            const bikes = bikesAt(station, 0);
            rows.push([
                meta.replayDate, overviewTimelineSlots[currentTimelineIndex], station.district, station.name, station.pattern,
                bikes == null ? '資料不足' : bikes, station.total,
                warning.immediate ? `即時${warning.kind === 'shortage' ? '缺車' : '滿站'}` : `${warning.horizon}分鐘${warning.kind === 'shortage' ? '缺車' : '滿站'}`,
                pct(warning.ratio), warning.immediate ? '事件定義 10.0%' : pct(warning.threshold)
            ]);
        });
        const csv = '\uFEFF' + rows.map(row => row.map(value => `"${String(value).replaceAll('"', '""')}"`).join(',')).join('\r\n');
        const url = URL.createObjectURL(new Blob([csv], { type: 'text/csv;charset=utf-8' }));
        const link = document.createElement('a'); link.href = url;
        link.download = `YouBike監理日報_${meta.replayDate}_${overviewTimelineSlots[currentTimelineIndex].replace(':', '')}.csv`;
        link.click(); URL.revokeObjectURL(url);
    };

    const originalUpdatePredictiveTime = window.updatePredictiveTime;
    window.updatePredictiveTime = function (value) {
        originalUpdatePredictiveTime(value);
        renderAlerts(); renderRelayCandidates(); buildDispatchCandidate();
    };

    currentPatternFilter = 'ALL';
    tablePatternFilter = 'ALL';
    gisPatternFilter = 'ALL';

    // 公開更新介面：讓 liveMode.js 在切換到 MME / 離線備援後，能重算站數、分群、
    // 行政區、來源與時間（否則沿用 appData 靜態 1576/424 等舊資訊）。
    // 可帶入來源覆寫（MME 模式傳入 API 回應的日期／來源字樣，取代 appData 靜態值）。
    window.hydrateMetadata = hydrateMetadata;
    window.populateFilters = populateFilters;

    window.addEventListener('load', () => {
        hydrateMetadata(); populateFilters();
        handleStationSearch(''); renderMarkers();
        if (stationData.length) selectStationForPanel(stationData[0].id);
    });
}());
