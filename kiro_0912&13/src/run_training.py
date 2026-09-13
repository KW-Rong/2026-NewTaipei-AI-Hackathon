# -*- coding: utf-8 -*-
"""送出並監控四個 XGBoost Script Mode Training Job (bike-30/dock-30/bike-60/dock-60)。

安全原則：不寫入任何憑證；僅用 hackathon-team3 profile。所有輸出限於
youbike-xgboost/runs/{RUN_ID}/ 之下。送出前先查同名 Job；逾時先查不重送。
共用本次 Processing 產生的 feature_spec 與 train/validation Parquet；6 月不進訓練。
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


def clients(cfg):
    sess = boto3.Session(profile_name=cfg["aws"]["profile"], region_name=cfg["aws"]["region"])
    bc = Config(connect_timeout=10, read_timeout=60, retries={"max_attempts": 5, "mode": "standard"})
    return sess, sess.client("sagemaker", config=bc), sess.client("s3", config=bc)


def training_job_exists(sm, name):
    try:
        return sm.describe_training_job(TrainingJobName=name)
    except ClientError as e:
        if e.response.get("Error", {}).get("Code") in ("ValidationException", "ResourceNotFound"):
            return None
        raise


def record_state(patch):
    art = os.path.join(ROOT, "artifacts", "run_state.json")
    state = {}
    if os.path.exists(art):
        with open(art, "r", encoding="utf-8") as f:
            state = json.load(f)
    tj = state.get("training_jobs", {})
    tj.update(patch)
    state["training_jobs"] = tj
    with open(art, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def upload_source(s3, bucket, code_prefix):
    """把訓練腳本打包成 sourcedir.tar.gz 上傳 (Script Mode 需要 tar.gz)。回傳 S3 URI。"""
    import tarfile
    import tempfile
    local = os.path.join(HERE, "training", "train_xgb.py")
    tmp = os.path.join(tempfile.gettempdir(), "sourcedir.tar.gz")
    with tarfile.open(tmp, "w:gz") as tf:
        tf.add(local, arcname="train_xgb.py")
    key = code_prefix + "sourcedir.tar.gz"
    s3.upload_file(tmp, bucket, key, ExtraArgs={"ServerSideEncryption": "AES256"})
    try:
        os.remove(tmp)
    except OSError:
        pass
    return f"s3://{bucket}/{key}"


def build_training_request(cfg, target, code_uri, suffix=""):
    """用 XGBoost 容器 (Script Mode) 送訓練。以 hyperparameters 傳入 sagemaker_program 等。"""
    from sagemaker import image_uris
    run_id = cfg["run_id"]
    bucket = cfg["aws"]["bucket"]
    role = cfg["aws"]["execution_role_arn"]
    s3 = cfg["s3"]
    tgt = cfg["targets"][target]
    hp = cfg["hyperparameters"]
    lr = cfg["label_rules"]

    image = image_uris.retrieve(framework="xgboost", region=cfg["aws"]["region"],
                                version=cfg["compute"]["xgboost_framework_version"])
    job_name = f"{run_id}-{target}" + (f"-{suffix}" if suffix else "")

    pp = s3["processed_prefix"]
    train_uri = f"s3://{bucket}/{pp}train.parquet"        # 只有 1-4 月
    valid_uri = f"s3://{bucket}/{pp}validation.parquet"   # 只有 5 月
    spec_uri = f"s3://{bucket}/{pp}feature_spec.json"     # 特徵規格另行指定
    # 註：6 月 test.parquet 不指定給任何 channel，確保不進 Training Job
    out_path = f"s3://{bucket}/{s3['models_prefix']}{target}/"

    # Script Mode：用 SageMaker XGBoost 容器的通用 entrypoint (sagemaker_program)
    hyperparameters = {
        "sagemaker_program": "train_xgb.py",
        "sagemaker_submit_directory": code_uri,
        "target": target,
        "label-col": tgt["label_col"],
        "horizon": str(tgt["horizon_min"]),
        "num-round": str(hp["num_round"]),
        "eta": str(hp["eta"]),
        "max-depth": str(hp["max_depth"]),
        "min-child-weight": str(hp["min_child_weight"]),
        "subsample": str(hp["subsample"]),
        "colsample-bytree": str(hp["colsample_bytree"]),
        "alpha": str(hp["alpha"]),
        "reg-lambda": str(hp["lambda"]),
        "max-bin": str(hp["max_bin"]),
        "early-stopping-rounds": str(hp["early_stopping_rounds"]),
        "seed": str(hp["seed"]),
        "low-ratio-threshold": str(lr["low_ratio_threshold"]),
        "low-ratio-weight": str(lr["low_ratio_sample_weight"]),
    }

    def chan(uri):
        # 用精確物件 key (S3Prefix 指到單一檔案)，避免下載同 prefix 其他檔 (含 6 月 test)
        return {"DataSource": {"S3DataSource": {
            "S3DataType": "S3Prefix", "S3Uri": uri, "S3DataDistributionType": "FullyReplicated"}},
            "InputMode": "File"}

    req = {
        "TrainingJobName": job_name,
        "RoleArn": role,
        "AlgorithmSpecification": {"TrainingImage": image, "TrainingInputMode": "File"},
        "HyperParameters": hyperparameters,
        "InputDataConfig": [
            {"ChannelName": "train", **chan(train_uri)},
            {"ChannelName": "validation", **chan(valid_uri)},
            {"ChannelName": "spec", **chan(spec_uri)},
        ],
        "OutputDataConfig": {"S3OutputPath": out_path},
        "ResourceConfig": {
            "InstanceType": cfg["compute"]["training_instance_type"],
            "InstanceCount": cfg["compute"]["training_instance_count"],
            "VolumeSizeInGB": cfg["compute"]["training_volume_gb"],
        },
        "StoppingCondition": {"MaxRuntimeInSeconds": 14400},
    }
    return job_name, req, image


TARGETS = ["bike-30", "dock-30", "bike-60", "dock-60"]


def submit_one(cfg, sm, s3, target, code_uri, suffix=""):
    job_name = f"{cfg['run_id']}-{target}" + (f"-{suffix}" if suffix else "")
    existing = training_job_exists(sm, job_name)
    if existing is not None:
        st = existing["TrainingJobStatus"]
        print(f"[skip-create] {job_name} 已存在 status={st}")
        record_state({target: {"name": job_name, "status": st,
                               "arn": existing.get("TrainingJobArn")}})
        return job_name
    job_name, req, image = build_training_request(cfg, target, code_uri, suffix=suffix)
    print(f"[submit] {job_name} image={image}")
    record_state({target: {"name": job_name, "status": "SUBMITTING"}})
    try:
        resp = sm.create_training_job(**req)
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code")
        msg = e.response.get("Error", {}).get("Message", "")
        print(f"[CREATE_ERROR] {target} {code}: {msg}", file=sys.stderr)
        chk = training_job_exists(sm, job_name)
        if chk is not None:
            print(f"[recovered] {job_name} 實際已建立")
            record_state({target: {"name": job_name, "status": chk["TrainingJobStatus"]}})
            return job_name
        record_state({target: {"name": job_name, "status": "CREATE_FAILED", "error_code": code}})
        raise
    record_state({target: {"name": job_name, "status": "InProgress",
                          "arn": resp.get("TrainingJobArn")}})
    print(f"[created] {job_name} ARN={resp.get('TrainingJobArn')}")
    return job_name


def monitor_all(sm, names, poll=45):
    terminal = {"Completed", "Failed", "Stopped"}
    done = {}
    while len(done) < len(names):
        for n in names:
            if n in done:
                continue
            d = sm.describe_training_job(TrainingJobName=n)
            st = d["TrainingJobStatus"]
            if st in terminal:
                done[n] = d
                print(f"[{n}] -> {st}", flush=True)
        if len(done) < len(names):
            time.sleep(poll)
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--action", choices=["submit", "monitor"], default="submit")
    ap.add_argument("--targets", default="", help="逗號分隔，只處理這些 target；空=全部四個")
    ap.add_argument("--suffix", default="", help="job 名稱後綴 (重送用新名稱)")
    args = ap.parse_args()
    cfg = load_cfg()
    sess, sm, s3 = clients(cfg)
    sel_targets = [t.strip() for t in args.targets.split(",") if t.strip()] or TARGETS
    sfx = args.suffix
    names = [f"{cfg['run_id']}-{t}" + (f"-{sfx}" if sfx else "") for t in sel_targets]

    if args.action == "monitor":
        done = monitor_all(sm, names)
        summary = {}
        for t, n in zip(sel_targets, names):
            d = done[n]
            summary[t] = {
                "name": n, "status": d["TrainingJobStatus"],
                "failure_reason": d.get("FailureReason"),
                "model_artifacts": d.get("ModelArtifacts", {}).get("S3ModelArtifacts"),
                "arn": d.get("TrainingJobArn"),
                "start": str(d.get("TrainingStartTime")), "end": str(d.get("TrainingEndTime")),
            }
            record_state({t: summary[t]})
        json.dump(summary, open(os.path.join(ROOT, "artifacts", "training_summary.json"),
                                "w", encoding="utf-8"), ensure_ascii=False, indent=2, default=str)
        print("TRAIN_MONITOR_DONE")
        for t in sel_targets:
            print(t, summary[t]["status"], summary[t]["model_artifacts"])
        return

    # submit：上傳一次程式碼，選定的 target 提交
    code_uri = upload_source(s3, cfg["aws"]["bucket"], cfg["s3"]["code_prefix"])
    print("[code]", code_uri)
    submitted = []
    for t in sel_targets:
        submitted.append(submit_one(cfg, sm, s3, t, code_uri, suffix=sfx))
    print("TRAIN_SUBMITTED", ",".join(submitted))


if __name__ == "__main__":
    main()
