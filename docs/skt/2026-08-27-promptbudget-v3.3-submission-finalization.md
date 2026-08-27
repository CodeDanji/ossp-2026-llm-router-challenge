# PromptBudget v3.3 제출 Finalization

## Frozen 제출 대상

- Artifact: `build/hash-regex-tail-guard/final-artifact.json`
- SHA-256: `c60d38ce2df670e6206689392389b66d89f810936969bcf47c2a0f29b86b88ce`
- Manifest: `build/hash-regex-tail-guard/manifest.json`
- Artifact type: `ossp-hash-regex-tail-guard-v1`

이 artifact의 학습 결과, tail guard, score/cost head 및 tier safety ratio는 제출 finalization에서 변경하지 않는다.

## Runtime 경로

`container/Dockerfile`은 `src`의 bundled policy resource, `baselines/hash_regex.py`, frozen artifact와 manifest를 포함한다. 컨테이너는 다음 경로로 실행된다.

```text
/opt/router/entrypoint.py
  -> /opt/router/baselines/hash_regex.py
  -> /opt/router/build/hash-regex-tail-guard/final-artifact.json
```

Fast와 Balanced는 frozen batch allocator를, Premium은 같은 complete route에서 `fill_ax31_upgrades`를 적용한다.

## 제출 순서

1. 이 문서를 포함한 코드 commit에서 `linux/arm64` 이미지를 build/push하고 immutable image digest를 기록한다.
2. `submission-ossp-skt.json`에 해당 code commit SHA와 image digest를 기록한다.
3. metadata-only commit을 push하고, 그 commit의 GitHub snapshot URL을 결과보고서의 프로젝트 등록 URL로 사용한다.

Train/Dev/full evaluation이나 v4 실험은 이 finalization 과정에서 실행하지 않는다.
