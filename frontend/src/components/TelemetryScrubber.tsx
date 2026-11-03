import { useMemo } from "react";
import { CartesianGrid, Legend, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { AnomalyEvent, HistoryPoint, RuntimeStatus } from "../types";

interface Props {
  history: HistoryPoint[]; // live stream history
  anomalies: AnomalyEvent[];
  status: RuntimeStatus | null;
  scrub: { frames: HistoryPoint[]; pos: number } | null; // frozen ring-buffer snapshot while replaying
  onEnterScrub: (t?: number) => void;
  onScrubPos: (pos: number) => void;
  onLive: () => void;
}

const MAX_CHART_POINTS = 160;

/** Peak-preserving decimation so the chart redraws cheaply without hiding short spikes. */
function downsample(points: HistoryPoint[]): HistoryPoint[] {
  const stride = Math.ceil(points.length / MAX_CHART_POINTS);
  if (stride <= 1) return points;
  const out: HistoryPoint[] = [];
  for (let i = 0; i < points.length; i += stride) {
    const bucket = points.slice(i, i + stride);
    const last = bucket[bucket.length - 1];
    out.push({
      ...last,
      joint_error: Math.max(...bucket.map((p) => p.joint_error)),
      chunk_variance: Math.max(...bucket.map((p) => p.chunk_variance)),
      max_load: Math.max(...bucket.map((p) => p.max_load)),
    });
  }
  return out;
}

export default function TelemetryScrubber({ history, anomalies, status, scrub, onEnterScrub, onScrubPos, onLive }: Props) {
  const data = useMemo(() => (scrub ? scrub.frames : downsample(history)), [scrub, history]);
  const tEnd = data.length ? data[data.length - 1].t : 0;
  const tStart = data.length ? data[0].t : 0;
  const cursor = scrub ? scrub.frames[scrub.pos] : null;
  const thr = status?.thresholds;

  const markers = useMemo(
    () => anomalies.filter((a) => a.timestamp >= tStart && a.timestamp <= tEnd),
    [anomalies, tStart, tEnd],
  );

  const handleClick = (e: { activeLabel?: string | number } | null) => {
    if (!e || e.activeLabel === undefined) return;
    const t = Number(e.activeLabel);
    if (!scrub) {
      onEnterScrub(t);
      return;
    }
    let best = 0;
    scrub.frames.forEach((f, i) => {
      if (Math.abs(f.t - t) < Math.abs(scrub.frames[best].t - t)) best = i;
    });
    onScrubPos(best);
  };

  return (
    <div className="scrubber">
      <div className="scrubber-head">
        <h3>Telemetry</h3>
        <span className="muted">
          joint following error, action-chunk variance and peak servo load. Click the chart to rewind the ring buffer (~10 s).
        </span>
        <span className="spacer" />
        {scrub ? (
          <button className="btn small live-btn" onClick={onLive}>● Back to LIVE</button>
        ) : (
          <button className="btn small" onClick={() => onEnterScrub()} disabled={!history.length}>⏪ Rewind</button>
        )}
      </div>
      <div className="chart">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data} margin={{ top: 6, right: 8, left: -12, bottom: 0 }} onClick={handleClick}>
            <CartesianGrid stroke="#1f2a37" strokeDasharray="3 3" />
            <XAxis
              dataKey="t"
              type="number"
              domain={["dataMin", "dataMax"]}
              tickFormatter={(t: number) => `${(t - tEnd).toFixed(0)}s`}
              stroke="#6b7a8c"
              fontSize={11}
            />
            <YAxis yAxisId="deg" stroke="#6b7a8c" fontSize={11} domain={[0, (max: number) => Math.max(40, Math.ceil(max))]} />
            <YAxis yAxisId="var" orientation="right" stroke="#6b7a8c" fontSize={11} domain={[0, (max: number) => Math.max(40, Math.ceil(max))]} />
            <Tooltip
              contentStyle={{ background: "#0f1620", border: "1px solid #253244", fontSize: 12 }}
              labelFormatter={(t) => `t = ${(Number(t) - tEnd).toFixed(2)} s`}
              formatter={(v: number, name: string) => [v.toFixed(2), name]}
            />
            <Legend wrapperStyle={{ fontSize: 11 }} />
            <Line yAxisId="deg" type="linear" dataKey="joint_error" name="joint error (deg)" stroke="#ff7b54" dot={false} isAnimationActive={false} strokeWidth={1.6} />
            <Line yAxisId="var" type="linear" dataKey="chunk_variance" name="chunk variance" stroke="#b18cff" dot={false} isAnimationActive={false} strokeWidth={1.6} />
            <Line yAxisId="deg" type="linear" dataKey="max_load" name="max load (%)" stroke="#4aa3ff" dot={false} isAnimationActive={false} strokeWidth={1.2} strokeOpacity={0.8} />
            {thr && <ReferenceLine yAxisId="deg" y={thr.joint_error_deg} stroke="#ff7b54" strokeDasharray="5 4" ifOverflow="extendDomain" />}
            {thr && <ReferenceLine yAxisId="var" y={thr.chunk_variance} stroke="#b18cff" strokeDasharray="5 4" ifOverflow="extendDomain" />}
            {markers.map((a) => (
              <ReferenceLine
                key={a.timestamp}
                yAxisId="deg"
                x={a.timestamp}
                stroke="#ff4d4d"
                strokeWidth={2}
                label={{ value: "⚠", position: "insideTop", fill: "#ff4d4d", fontSize: 14 }}
              />
            ))}
            {cursor && <ReferenceLine yAxisId="deg" x={cursor.t} stroke="#ffffff" strokeWidth={2} />}
          </LineChart>
        </ResponsiveContainer>
      </div>
      {scrub && (
        <div className="scrub-bar">
          <input
            type="range"
            min={0}
            max={Math.max(0, scrub.frames.length - 1)}
            value={scrub.pos}
            onChange={(e) => onScrubPos(Number(e.target.value))}
          />
          {cursor && (
            <span className="mono">
              frame {cursor.index} · {(cursor.t - tEnd).toFixed(2)} s · {cursor.phase ?? ""} · err {cursor.joint_error.toFixed(1)}° · var{" "}
              {cursor.chunk_variance.toFixed(1)}
            </span>
          )}
        </div>
      )}
    </div>
  );
}
