"""
將 YouBike 資料集（訓練/驗證/測試）上傳至私有、加密的 Amazon S3 bucket。

安全原則：
- 不在程式碼內寫入任何 AWS Access Key / Secret Key / Session Token。
- 僅透過本機 AWS profile (hackathon-team3) 由 boto3 讀取憑證。
- 不印出任何憑證；STS 僅顯示 Account / ARN / Region。
- 保留原始檔名，不修改 CSV 內容。
- 所有物件使用 SSE-S3 (AES256) 加密；bucket 啟用 Block Public Access。

本程式只負責上傳原始資料，不做特徵工程、不訓練模型、不建立 Endpoint。
"""
import datetime
import hashlib  # noqa: F401  (保留給未來校驗；目前以大小驗證)
import json
import os
import sys

import boto3
from boto3.s3.transfer import TransferConfig
from botocore.config import Config
from botocore.exceptions import ClientError, BotoCoreError, ProfileNotFound

# ---- 固定設定（不含任何憑證）----
PROFILE_NAME = "hackathon-team3"
REGION = "us-west-2"
PREFIX_ROOT = "youbike-xgboost/raw"
SSE_ALGO = "AES256"

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, "artifacts")
MANIFEST_PATH = os.path.join(ARTIFACTS_DIR, "s3_upload_manifest.json")

# split -> (本機資料夾, S3 子前綴)
SPLITS = {
    "train": ("訓練集_一到四月", "train"),
    "validation": ("驗證集_五月", "validation"),
    "test": ("測試集_六月", "test"),
}

MB = 1024 * 1024
TRANSFER_CONFIG = TransferConfig(
    multipart_threshold=64 * MB,
    multipart_chunksize=64 * MB,
    max_concurrency=4,
    use_threads=True,
)


def fail(msg, code=1):
    print(f"[STOP] {msg}")
    sys.exit(code)


def report_access_denied(api, err, current_file, uploaded):
    code = err.response.get("Error", {}).get("Code") if isinstance(err, ClientError) else type(err).__name__
    print("\n==== ACCESS DENIED / 權限錯誤，立即停止 ====")
    print(f"失敗的 AWS API : {api}")
    print(f"AWS 錯誤代碼   : {code}")
    print(f"當下處理檔案   : {current_file}")
    print(f"已成功上傳     : {uploaded if uploaded else '（無）'}")
    print("需要人工開啟的最小 S3 權限（請勿開放 s3:*）：")
    print("  s3:CreateBucket, s3:GetBucketLocation, s3:ListBucket,")
    print("  s3:PutObject, s3:GetObject, s3:AbortMultipartUpload,")
    print("  s3:ListBucketMultipartUploads, s3:ListMultipartUploadParts,")
    print("  s3:PutBucketPublicAccessBlock, s3:PutEncryptionConfiguration")
    sys.exit(2)


def get_session():
    try:
        return boto3.Session(profile_name=PROFILE_NAME, region_name=REGION)
    except ProfileNotFound:
        fail(f"找不到 AWS profile '{PROFILE_NAME}'，請確認 ~/.aws 設定（勿將憑證寫入專案）。")


def whoami(session):
    sts = session.client("sts")
    try:
        ident = sts.get_caller_identity()
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code")
        if code in ("ExpiredToken", "ExpiredTokenException", "InvalidClientTokenId"):
            fail("Workshop 臨時憑證已過期，請更新 ~/.aws 的 hackathon-team3 profile（勿重建 profile、勿寫入專案）。")
        raise
    print("=== STS get_caller_identity ===")
    print("Account:", ident["Account"])
    print("ARN    :", ident["Arn"])
    print("Region :", session.region_name)
    return ident["Account"]


def ensure_bucket(s3, bucket, account_id):
    """檢查 bucket 是否存在且屬於本帳號；不存在才建立。回傳 True=已就緒。"""
    exists_and_owned = False
    try:
        s3.head_bucket(Bucket=bucket)
        exists_and_owned = True
        print(f"[bucket] 已存在且可存取，沿用：{bucket}")
    except ClientError as e:
        err = e.response.get("Error", {})
        status = err.get("Code")
        if status in ("403", "AccessDenied"):
            report_access_denied("s3:HeadBucket/s3:ListBucket", e, bucket, [])
        elif status in ("404", "NoSuchBucket"):
            exists_and_owned = False
        else:
            # 其他狀況（例如 301）視為不可安全沿用
            fail(f"檢查 bucket 時發生非預期錯誤：{status}")

    if not exists_and_owned:
        print(f"[bucket] 不存在，建立中：{bucket}")
        try:
            s3.create_bucket(
                Bucket=bucket,
                CreateBucketConfiguration={"LocationConstraint": REGION},
            )
        except ClientError as e:
            code = e.response.get("Error", {}).get("Code")
            if code in ("AccessDenied",):
                report_access_denied("s3:CreateBucket", e, bucket, [])
            elif code in ("BucketAlreadyOwnedByYou",):
                print("[bucket] 已由本帳號擁有，沿用。")
            elif code in ("BucketAlreadyExists",):
                fail("bucket 名稱已被其他帳號使用，停止（不覆蓋他人資料）。")
            else:
                raise

    # 私有化：Block Public Access
    try:
        s3.put_public_access_block(
            Bucket=bucket,
            PublicAccessBlockConfiguration={
                "BlockPublicAcls": True,
                "IgnorePublicAcls": True,
                "BlockPublicPolicy": True,
                "RestrictPublicBuckets": True,
            },
        )
        print("[bucket] 已啟用 Block Public Access")
    except ClientError as e:
        if e.response.get("Error", {}).get("Code") == "AccessDenied":
            report_access_denied("s3:PutBucketPublicAccessBlock", e, bucket, [])
        raise

    # 預設加密：SSE-S3 AES256
    try:
        s3.put_bucket_encryption(
            Bucket=bucket,
            ServerSideEncryptionConfiguration={
                "Rules": [
                    {"ApplyServerSideEncryptionByDefault": {"SSEAlgorithm": "AES256"}}
                ]
            },
        )
        print("[bucket] 已設定預設加密 SSE-S3 (AES256)")
    except ClientError as e:
        if e.response.get("Error", {}).get("Code") == "AccessDenied":
            report_access_denied("s3:PutEncryptionConfiguration", e, bucket, [])
        raise


def collect_files():
    """回傳 [(split, s3_sub, local_path, filename)]，並確認數量。"""
    expected = {"train": 4, "validation": 1, "test": 1}
    items = []
    for split, (folder, s3_sub) in SPLITS.items():
        dpath = os.path.join(PROJECT_ROOT, folder)
        if not os.path.isdir(dpath):
            fail(f"資料夾不存在：{folder}")
        csvs = sorted(f for f in os.listdir(dpath) if f.lower().endswith(".csv"))
        if len(csvs) != expected[split]:
            fail(f"{split} CSV 數量不符：預期 {expected[split]}，實際 {len(csvs)}")
        for fname in csvs:
            items.append((split, s3_sub, os.path.join(dpath, fname), fname))
    return items


def main():
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    session = get_session()
    account_id = whoami(session)

    bucket = f"sagemaker-ntpc-youbike-{account_id}-us-west-2"
    s3_root = f"s3://{bucket}/{PREFIX_ROOT}/"

    boto_cfg = Config(retries={"max_attempts": 5, "mode": "standard"})
    s3 = session.client("s3", config=boto_cfg)

    ensure_bucket(s3, bucket, account_id)

    files = collect_files()

    manifest = {
        "aws_region": REGION,
        "bucket": bucket,
        "s3_root": s3_root,
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "objects": [],
    }
    uploaded_names = []
    counts = {"train": 0, "validation": 0, "test": 0}

    for split, s3_sub, local_path, fname in files:
        key = f"{PREFIX_ROOT}/{s3_sub}/{fname}"
        local_size = os.path.getsize(local_path)
        print(f"\n[upload] {split} -> s3://{bucket}/{key} ({local_size} bytes)")

        # 上傳
        try:
            s3.upload_file(
                Filename=local_path,
                Bucket=bucket,
                Key=key,
                ExtraArgs={"ServerSideEncryption": SSE_ALGO},
                Config=TRANSFER_CONFIG,
            )
        except ClientError as e:
            if e.response.get("Error", {}).get("Code") == "AccessDenied":
                report_access_denied("s3:PutObject", e, fname, uploaded_names)
            raise
        except BotoCoreError as e:
            fail(f"上傳 {fname} 失敗（網路/傳輸錯誤）：{type(e).__name__}")

        # head_object 驗證
        try:
            head = s3.head_object(Bucket=bucket, Key=key)
        except ClientError as e:
            if e.response.get("Error", {}).get("Code") == "AccessDenied":
                report_access_denied("s3:GetObject/HeadObject", e, fname, uploaded_names)
            raise
        s3_size = head["ContentLength"]
        ok = (s3_size == local_size)
        status = "SUCCESS" if ok else "FAILED_SIZE_MISMATCH"

        obj = {
            "split": split,
            "local_filename": fname,
            "size_bytes": local_size,
            "s3_content_length": s3_size,
            "s3_key": key,
            "s3_uri": f"s3://{bucket}/{key}",
            "sse": SSE_ALGO,
            "status": status,
            "uploaded_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }
        manifest["objects"].append(obj)
        print(f"  local={local_size}  s3={s3_size}  -> {status}")

        if not ok:
            with open(MANIFEST_PATH, "w", encoding="utf-8") as mf:
                json.dump(manifest, mf, ensure_ascii=False, indent=2)
            fail(f"{fname} 大小不一致（local={local_size}, s3={s3_size}），視為失敗並停止。")

        uploaded_names.append(fname)
        counts[split] += 1

    with open(MANIFEST_PATH, "w", encoding="utf-8") as mf:
        json.dump(manifest, mf, ensure_ascii=False, indent=2)

    print("\n=== UPLOAD SUMMARY ===")
    print("bucket        :", bucket)
    print("s3_root       :", s3_root)
    print("train uploaded:", counts["train"])
    print("valid uploaded:", counts["validation"])
    print("test  uploaded:", counts["test"])
    print("manifest      :", MANIFEST_PATH)
    print("ALL_DONE")


if __name__ == "__main__":
    main()
