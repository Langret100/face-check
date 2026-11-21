# v18 거울 반전 · AvatarSample_A 적용

- 카메라 미리보기와 3D 렌더 캔버스 모두 좌우 반전해 거울 방식으로 표시
- 기존 `HairSample_Female` 제거, 업로드된 `AvatarSample_A` 사용
- VRM 내부 눈/입/감정 15종, 양쪽 팔 관절 구조 확인
- 사람형은 머리/가슴 뼈 기준 상반신 구도. 모바일 세로는 확대·축소 없이 고정
- 팔 자세 변경 시 어깨 관절에 소폭 추가 움직임 반영; Worker 몸 AI 실패 시 기본 추적 방식으로 재시도
- **라이선스:** AvatarSample_A는 CC0가 아님. `ASSET_LICENSES.md` 확인
- **검증 제한:** 문법·구조·가상 방향 검사는 했으나 실기기 60fps, 실제 얼굴/팔 추적은 미검증

# AI 표정 친구 v17 — AI 추론 Worker 분리

- 기존 Mr Squirrel·CC0 사람형 VRM, 선택 UI, 고정 화면 및 TTS/음성 변조 보존
- MediaPipe CPU 얼굴/몸 추론을 별도의 모듈 Web Worker에서 실행 시도
- Worker 초기화 불가/오류 시 기존 메인 스레드 방식 자동 전환
- 캐릭터는 AI 로딩 이전에 우선 표시; 얼굴 모델이 준비되면 카메라 추적 시작, 팔 모델은 뒤에서 로딩
- ImageBitmap 프레임 1개만 처리 중일 때 유지하여 큐 적체 최소화
- 모델·WASM·JavaScript GitHub Pages 자체 호스팅, 방문자 브라우저의 외부 CDN 요청 없음
- GitHub Actions: npm install 후 npm run build, 배포 폴더 dist

## 확인 필요
Worker에서 MediaPipe의 OffscreenCanvas 입력 지원은 배포 대상 브라우저에 따라 차이가 있을 수 있습니다. 이 경우 메인 스레드 추론 방식으로 자동 전환합니다. 실기기 FPS와 초기 로딩 시간은 아직 측정되지 않았습니다. 60fps는 목표이지 보장 수치가 아닙니다.
