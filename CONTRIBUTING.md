# 参与开发

## 本地验证

```sh
python -m venv .venv
.venv/bin/python -m pip install -r requirements-dev.txt
make build
make test
make format-check
make lint typecheck
```

每次提交只处理一个明确行为，并按上述顺序验证。测试使用小型合成输入，
默认不访问网络、下载模型或依赖 GPU。API 变化应同步更新完整契约快照，
并解释兼容性影响。版本号只增加补丁位，提交消息采用 conventional commits。

## 数据与结果

不要提交个人录音、凭据、大型模型或不可再分发的数据。示例与测试结果
只能说明所覆盖的行为，不能代替真实模型基准或用户研究。
