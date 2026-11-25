# `릴프로필` (`프로필`) — 하이브리드 클라우드 세션 저장 상태 (`2026-11-25`)

## 1. 정본 저장소 및 버전 상태
- **프로젝트 공식 명칭**: **`릴프로필`** (약칭 **`프로필`**, 하위 호환 `모델생성`)
- **GitHub 원격 저장소**: `https://github.com/domautube-beep/model-generation-engine` (`main`)
- **현재 정본 버전**: **`v1.0.5`** (`정본 v1.0.0` 디폴트 베이스라인 + `v0.1.0 ~ v0.9.5` 아카이브 분리 + 자동 버전가드 `scripts/version_ctl.py`)
- **로컬 경로**: `/Volumes/DOMA/Project/릴프로필` (`/Volumes/DOMA/Project/모델생성`, `~/Projects/릴프로필`, `~/Projects/프로필`)
- **독립 실행 전체 패키지**:
  - 로컬: `/Volumes/DOMA/Project/릴프로필_정본_v1.0.0_전체패키지.zip` (`136.8 MB`, `v1.0.5` 내장)
  - Google Drive 미러: `Antigravity_Assets/모델생성/releases/릴프로필_정본_v1.0.0_전체패키지.zip`

## 2. 확정된 디폴트 촬영·마스터링 규격 (`v1.0.5`)
1. **기본 `50:50` 인물·배경 여백 밸런스 (`balanced_50_50`, 말사전 `생-035`)**:
   - 인물로 프레임을 꽉 채우지 않고 **인물 약 `45~50%` : 주변 건축 여백·헤드룸 약 `50~55%`**를 기본으로 유지.
   - 사용자가 *"더 클로즈업해줘"* 등 명시적으로 요청할 때만 `closeup` 모드로 전환.
2. **기본 `f/1.8~f/2.0` 개방 아웃포커싱 & 3단 공간 깊이감 (`Z-Depth`)**:
   - 배경 디테일을 과잉 묘사하지 않고 초점은 눈·얼굴에만 칼핀으로 맞춘 뒤 후경을 부드러운 광학 보케로 날려 깊이감을 살림.
3. **`Zero-Touch 4K` 무변형 직통 모드 (`zero_touch_4k` + `optical_fullframe`, 말사전 `생-034` · `사례-01285ef`)**:
   - 로컬 얼굴 국소 필터 3종(`증명사진 피부색 역주입 OFF`, `MatteSkin 하이라이트 압축 OFF`, `눈·코·입 국소 샤프닝 & 부위별 GAN 합성 OFF`) 완전 차단 → 4K 저장 시 처음 생성된 조명·색감·질감·생김새 변형 `0.0px` (`max_abs_pixel_diff_vs_raw == 0.0`).

## 3. 2-Tier 하이브리드 클라우드 동기화 경로
- **Tier 1 (iCloud `< 50KB` 경량 스킬 타워)**:
  - `~/Library/Mobile Documents/com~apple~CloudDocs/Antigravity_Control_Tower/skills/reel-profile/SKILL.md`
  - `~/Library/Mobile Documents/com~apple~CloudDocs/Antigravity_Control_Tower/skills/model-generation-engine/SKILL.md`
- **Tier 2 (Google Drive 대용량 자산 볼트)**:
  - `~/Library/CloudStorage/GoogleDrive-*/내 드라이브/Antigravity_Assets/모델생성/`
    - `models/` (`yunet.onnx`, `sface.onnx`, `upscayl-bin`, `apple_vision_bin` 등)
    - `resources/subject_05/`, `subject_06/`, `subject_07/` (`gen0_origin.jpg` + `8dir_bank/*_master_8angle_turnaround_sheet.jpg`)
    - `resources/reference_moods_2026_11_24/` (레퍼런스 무드·포즈 사진 6종)
    - `outputs/2026-11-24_v43_3men_5050_bokeh_zerotouch4k/` (`v1.0.5` `Zero-Touch 4K` 결과물 및 `resume_batches/v43_flow_batch_s05..s07.json`)
    - `releases/릴프로필_정본_v1.0.0_전체패키지.zip`
