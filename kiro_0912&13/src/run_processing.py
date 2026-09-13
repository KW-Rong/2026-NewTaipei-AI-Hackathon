# -*- coding: utf-8 -*-
"""送出並監控 SageMaker Processing Job (共用特徵工程)。

安全原則：不寫入任何憑證；僅用 hackathon-team3 profile。所有輸出路徑限定於
youbike-xgboost/runs/{RUN_ID}/ 之下 (符合 execution role 的 PutObject 範圍)。
送出前先查同名 Job；若已存在則沿用監控，不重複建立。
"""
import argparse
import json
import os
import sys
import time

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CONFIG_PATH = os.path.join(ROOT, "config", "run_config.json")


def load_cfg():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def session_and_clients(cfg):
    sess = boto3.Session(profile_name=cfg["aws"]["profile"], region_name=cfg["aws"]["region"])
    bcfg = Config(connect_timeout=10, read_timeout=60, retries={"max_attempts": 5, "mode": "standard"})
    return sess, sess.client("sagemaker", config=bcfg), sess.client("s3", config=bcfg)


def job_exists(sm, name):
    try:
        d = sm.describe_processing_job(ProcessingJobName=name)
        return d
    except ClientError as e:
        if e.response.get("Error", {}).get("Code") in ("ValidationException", "ResourceNotFound"):
            return None
        raise


def upload_code(s3, bucket, code_prefix, local_path, key_name):
    key = code_prefix + key_name
    s3.upload_file(local_path, bucket, key, ExtraArgs={"ServerSideEncryption": "AES256"})
    return f"s3://{bucket}/{key}"


def build_processing_request(cfg, code_s3_uri, suffix=""):
    run_id = cfg["run_id"]
    bucket = cfg["aws"]["bucket"]
    role = cfg["aws"]["execution_role_arn"]
    s3 = cfg["s3"]
    job_name = f"{run_id}-proc" + (f"-{suffix}" if suffix else "")
    inputs = [
        {"InputName": "train", "S3Input": {
            "S3Uri": f"s3://{bucket}/{s3['raw_train_prefix']}",
            "LocalPath": "/opt/ml/processing/input/train",
            "S3DataType": "S3Prefix", "S3InputMode": "File",
            "S3DataDistributionType": "FullyReplicated"}},
        {"InputName": "validation", "S3Input": {
            "S3Uri": f"s3://{bucket}/{s3['raw_validation_prefix']}",
            "LocalPath": "/opt/ml/processing/input/validation",
            "S3DataType": "S3Prefix", "S3InputMode": "File",
            "S3DataDistributionType": "FullyReplicated"}},
        {"InputName": "test", "S3Input": {
            "S3Uri": f"s3://{bucket}/{s3['raw_test_prefix']}",
            "LocalPath": "/opt/ml/processing/input/test",
            "S3DataType": "S3Prefix", "S3InputMode": "File",
            "S3DataDistributionType": "FullyReplicated"}},
        {"InputName": "code", "S3Input": {
            "S3Uri": code_s3_uri,
            "LocalPath": "/opt/ml/processing/input/code",
            "S3DataType": "S3Prefix", "S3InputMode": "File",
            "S3DataDistributionType": "FullyReplicated"}},
    ]
    outputs = [
        {"OutputName": "processed", "S3Output": {
            "S3Uri": f"s3://{bucket}/{s3['processed_prefix']}",
            "LocalPath": "/opt/ml/processing/output",
            "S3UploadMode": "EndOfJob"}},
    ]
    # 使用 sklearn 內建容器跑我們的腳本 (含 pandas/numpy/scikit-learn)
    # 注意：sklearn 影像的 image_scope 無 "processing"，省略即可
    from sagemaker import image_uris
    image = image_uris.retrieve(framework="sklearn", region=cfg["aws"]["region"],
                                version="1.2-1")
    # entrypoint 先確保 pyarrow 可用 (寫 Parquet)，再執行特徵工程
    bootstrap = (
        "python3 -c 'import pyarrow' 2>/dev/null || pip install -q pyarrow; "
        "python3 /opt/ml/processing/input/code/feature_engineering.py --run-id " + run_id
    )
    req = {
        "ProcessingJobName": job_name,
        "RoleArn": role,
        "AppSpecification": {
            "ImageUri": image,
            "ContainerEntrypoint": ["/bin/bash", "-c", bootstrap],
        },
        "ProcessingInputs": inputs,
        "ProcessingOutputConfig": {"Outputs": outputs},
        "ProcessingResources": {"ClusterConfig": {
            "InstanceCount": cfg["compute"]["processing_instance_count"],
            "InstanceType": cfg["compute"]["processing_instance_type"],
            "VolumeSizeInGB": cfg["compute"]["processing_volume_gb"],
        }},
        "StoppingCondition": {"MaxRuntimeInSeconds": 5400},
    }
    return job_name, req, image


def record_state(cfg, patch):
    """把 Job 名稱等狀態寫入本機 artifacts (不含任何憑證)。"""
    art = os.path.join(ROOT, "artifacts", "run_state.json")
    state = {}
    if os.path.exists(art):
        with open(art, "r", encoding="utf-8") as f:
            state = json.load(f)
    state.setdefault("run_id", cfg["run_id"])
    state.update(patch)
    with open(art, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def monitor(sm, name, poll=30):
    terminal = {"Completed", "Failed", "Stopped"}
    while True:
        d = sm.describe_processing_job(ProcessingJobName=name)
        st = d["ProcessingJobStatus"]
        print(f"[{name}] {st}", flush=True)
        if st in terminal:
            return d
        time.sleep(poll)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--action", choices=["submit", "monitor"], default="submit")
    ap.add_argument("--suffix", default="", help="job 名稱後綴 (重跑用新名稱)")
    args = ap.parse_args()
    cfg = load_cfg()
    sess, sm, s3 = session_and_clients(cfg)
    bucket = cfg["aws"]["bucket"]
    code_prefix = cfg["s3"]["code_prefix"]

    job_name = f"{cfg['run_id']}-proc" + (f"-{args.suffix}" if args.suffix else "")

    if args.action == "monitor":
        d = monitor(sm, job_name)
        record_state(cfg, {"processing_job": {"name": job_name, "status": d["ProcessingJobStatus"],
                                              "arn": d.get("ProcessingJobArn")}})
        print("MONITOR_DONE", d["ProcessingJobStatus"])
        return

    # submit：先查是否已存在，存在則不重送
    existing = job_exists(sm, job_name)
    if existing is not None:
        print(f"[skip-create] Processing Job 已存在，狀態={existing['ProcessingJobStatus']}，改為監控")
        record_state(cfg, {"processing_job": {"name": job_name,
                                              "status": existing["ProcessingJobStatus"],
                                              "arn": existing.get("ProcessingJobArn")}})
        return

    # 上傳程式碼到 runs/{RUN_ID}/code/
    fe_local = os.path.join(HERE, "processing", "feature_engineering.py")
    code_uri = upload_code(s3, bucket, code_prefix, fe_local, "feature_engineering.py")
    print("[code] uploaded:", code_uri)

    job_name, req, image = build_processing_request(cfg, code_uri, suffix=args.suffix)
    print("[image]", image)
    print("[submit] creating processing job:", job_name)
    record_state(cfg, {"processing_job": {"name": job_name, "status": "SUBMITTING",
                                          "image": image, "code_uri": code_uri}})
    try:
        resp = sm.create_processing_job(**req)
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code")
        msg = e.response.get("Error", {}).get("Message", "")
        print(f"[CREATE_ERROR] {code}: {msg}", file=sys.stderr)
        # 逾時/不確定時：查是否其實已建立
        chk = job_exists(sm, job_name)
        if chk is not None:
            print("[recovered] Job 實際已建立，改為監控")
            record_state(cfg, {"processing_job": {"name": job_name,
                                                  "status": chk["ProcessingJobStatus"]}})
            return
        record_state(cfg, {"processing_job": {"name": job_name, "status": "CREATE_FAILED",
                                              "error_code": code}})
        sys.exit(3)
    arn = resp.get("ProcessingJobArn")
    print("[created] ARN:", arn)
    record_state(cfg, {"processing_job": {"name": job_name, "status": "InProgress", "arn": arn}})
    print("PROC_SUBMITTED", job_name)


if __name__ == "__main__":
    main()
