/**
 * Phase 1 병행 3종. **체험 화면과 기록 화면이 같은 arm을 가리키게** 하는 것이 목적이다.
 *
 * 최종 1종은 Phase 2 진입 시 정한다 — Phase 1의 변경(리랭커·하이브리드·의미 분해)이
 * 모델마다 다르게 들 수 있어서 지금 줄이면 그 상호작용을 못 본다.
 * 근거는 docs/experiments.md "Phase 1 병행 3종과 각자의 역할".
 */
export type ArmKey = "A" | "K" | "B";

export const ARMS: {
  key: ArmKey;
  model: string;
  role: string;
  /** Phase 0에서의 arm_id — 기록 페이지로 연결하는 키 */
  phase0ArmId: string;
  /** ⚠️ 질의에만 붙인다. 문서에는 붙이지 않는다. 빠뜨리면 MRR −16%. */
  queryPrefix: string;
}[] = [
  { key: "A", model: "Snowflake/snowflake-arctic-embed-l-v2.0", role: "주 후보 (다국어)", phase0ArmId: "E-7", queryPrefix: "query: " },
  { key: "K", model: "dragonkue/snowflake-arctic-embed-l-v2.0-ko", role: "한국어 상한선", phase0ArmId: "E-6", queryPrefix: "query: " },
  { key: "B", model: "BAAI/bge-m3", role: "다른 계열 대조", phase0ArmId: "E-2", queryPrefix: "" },
];

export const API_BASE = process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:8000";
