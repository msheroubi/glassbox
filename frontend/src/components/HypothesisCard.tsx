import CounterfactualCard from "./CounterfactualCard";
import type { HypothesisCheck } from "../types";

interface Props {
  check: HypothesisCheck;
  finalConfidence: number | undefined;
  finalCulprit: string | null | undefined;
  onSeekTime: (t: number) => void;
}

const RESULTS: Record<string, { cls: string; text: string }> = {
  supported: { cls: "ok", text: "hypothesis supported" },
  refuted: { cls: "warn", text: "hypothesis refuted" },
  inconclusive: { cls: "cache", text: "inconclusive" },
};

function pct(v: number): string {
  return `${Math.round(v * 100)}%`;
}

/** Muse Wonder proposed a hypothesis, chose a counterfactual to test it, and revised its breakdown with the result. */
export default function HypothesisCard({ check, finalConfidence, finalCulprit, onSeekTime }: Props) {
  const who = check.proposed_by === "local-fallback" ? "The rule engine" : "Muse Wonder";
  const running = check.status === "running";
  const result = check.result ? RESULTS[check.result] : null;
  const t = `${check.t > 0 ? "+" : ""}${check.t.toFixed(1)}`;

  return (
    <div className={`hypo-card ${check.status}`}>
      <div className="hypo-head">
        <b>🔬 {who} tested its own hypothesis</b>
        <span className="spacer" />
        {running && (
          <span className="chip cache">
            <span className="spinner tiny" /> testing
          </span>
        )}
        {result && <span className={`chip ${result.cls}`}>{result.text}</span>}
        {check.status === "failed" && <span className="chip warn">test could not run</span>}
      </div>

      <ol className="hypo-steps">
        <li>
          <span className="fact-k">Hypothesis</span>
          <p>{check.hypothesis}</p>
        </li>
        {check.prediction && (
          <li>
            <span className="fact-k">Prediction</span>
            <p>{check.prediction}</p>
          </li>
        )}
        <li>
          <span className="fact-k">Test it chose</span>
          <p>
            Remove the {check.remove_label} from what the policy saw at{" "}
            <button className="link mono" onClick={() => onSeekTime(check.t)} title="Jump the replay here">
              t = {t} s
            </button>{" "}
            and ask the policy again where it looks and reaches, at the joint state it had then.
          </p>
        </li>
        <li>
          <span className="fact-k">Result</span>
          <p>
            {running ? (
              <span className="muted">
                <span className="spinner tiny" /> {check.counterfactual?.status === "attending" ? "asking the policy…" : "Muse Image is editing the frame…"}
              </span>
            ) : (
              check.result_summary || check.counterfactual?.verdict
            )}
          </p>
        </li>
        {check.status === "ready" && finalConfidence !== undefined && (
          <li>
            <span className="fact-k">Breakdown revised</span>
            <p>
              confidence {pct(check.first_confidence)} → <b>{pct(finalConfidence)}</b>
              {(check.first_culprit ?? null) !== (finalCulprit ?? null) && (
                <>
                  , culprit {check.first_culprit ?? "none"} → <b>{finalCulprit ?? "none"}</b>
                </>
              )}
              <span className="muted"> · first reading: “{check.first_headline}”</span>
            </p>
          </li>
        )}
      </ol>

      {check.counterfactual && (
        <CounterfactualCard cf={check.counterfactual} title={`${who === "Muse Wonder" ? "Wonder's" : "Its"} test`} onShowFrame={() => onSeekTime(check.t)} />
      )}
    </div>
  );
}
