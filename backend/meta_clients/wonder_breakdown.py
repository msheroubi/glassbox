"""Muse Wonder (``muse-wonder-1.0``) forensic breakdown of a recorded failure.

Muse Spark gives the fast live triage from one keyframe. Once the run has been replayed offline,
Wonder gets the whole story: a filmstrip of what the policy saw with its attention and the SAM masks,
the attention share of every object over time for each attention map, the detected attention shifts,
the servo telemetry and Spark's triage. It answers three questions:

* At which stage did the failure start: perception (it saw the scene wrong), decision (it saw the
  scene right and still acted wrong) or actuation (the arm could not do what the policy asked)?
* What caused it, and what does the evidence over time show?
* Can GlassBox correct it by itself (an intervention on the policy's input or the trajectory), or does
  it need a human (clear the workspace, retrain)?

Then it tests its own hypothesis: the first reading proposes one more counterfactual of its choosing (remove
an object at a recorded moment) and what it predicts the test will show. GlassBox runs it (``replay.py``) and
a second turn (``conclude``) writes the final breakdown with the result as evidence.

If Wonder rejects the image the request is retried text-only; without the API (and no cached response)
a rule engine over the same evidence produces the breakdown, and proposes and judges the follow-up test.
"""

from __future__ import annotations

import copy
import dataclasses
import json
import logging
from dataclasses import dataclass, field

import numpy as np

from backend.config import Settings
from backend.schemas import FailureBreakdown, ReplayEvent, TimelineEntry
from backend.xai_engine.attribution import NEAR_DRAW_LIFT, NON_DISTRACTOR_LABELS
from backend.xai_engine.visualize import data_url

from .base import MetaAPIError, MetaClient, json_schema_format, output_text, user_message
from .spark_explainer import INTERVENTIONS, _parse_json

log = logging.getLogger(__name__)

STAGES = ["PERCEPTION", "DECISION", "ACTUATION", "NONE"]
AUTOMATIC = {"INPAINT_MASK_SUPPRESSION", "WAYPOINT_BYPASS", "RETRY_GRASP_OFFSET"}

BREAKDOWN_SCHEMA = {
    "type": "object",
    "properties": {
        "headline": {"type": "string", "description": "One sentence for the operator, max 20 words."},
        "failure_stage": {"type": "string", "enum": STAGES},
        "culprit_label": {"type": ["string", "null"], "description": "SAM label of the object behind the failure"},
        "perception_analysis": {
            "type": "string", "description": "2-3 sentences: what the policy perceived, and how it differs from the real scene.",
        },
        "root_cause": {"type": "string", "description": "2-4 sentences citing the numbers (shares, times)."},
        "timeline": {
            "type": "array",
            "maxItems": 6,
            "items": {
                "type": "object",
                "properties": {
                    "t": {"type": "number", "description": "seconds relative to the failure, <= 0"},
                    "event": {"type": "string"},
                    "evidence": {"type": "string"},
                },
                "required": ["t", "event", "evidence"],
            },
        },
        "counterfactual": {
            "type": "string",
            "description": "1-2 sentences: what the counterfactual test showed, and what the policy would most likely have done without the cause.",
        },
        "self_correctable": {"type": "boolean"},
        "needs_human": {"type": "boolean"},
        "human_action": {"type": "string", "description": "What a person must do, or an empty string."},
        "intervention_type": {"type": "string", "enum": INTERVENTIONS},
        "intervention_parameters": {
            "type": "object", "additionalProperties": {"type": ["number", "integer", "string"]},
        },
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "operator_explanation": {
            "type": "string", "description": "2-3 plain-language sentences a non-expert audience understands.",
        },
    },
    "required": [
        "headline", "failure_stage", "culprit_label", "perception_analysis", "root_cause", "timeline",
        "counterfactual", "self_correctable", "needs_human", "human_action", "intervention_type",
        "intervention_parameters", "confidence", "operator_explanation",
    ],
}

BRIEF = """You are the forensic analyst of GlassBox, an explainability layer for a LeRobot SO-101 arm whose
imitation-learning policy (ACT, trained on a few dozen demonstrations) just failed at the task: "{task}".
The thesis you are testing: most failures of learned robot policies are failures of PERCEPTION (the policy
thinks it sees something it doesn't, or something in view throws off its reading of the scene), not of
DECISION (it read the scene correctly and still chose wrong) or ACTUATION (the arm could not execute).
Only call it a perception failure if the attention evidence supports it.

The recorded run was replayed offline. The frames are the policy's own input (after any edits GlassBox made
to it). Maps over those frames:
{maps}
"act" is the policy that actually drove the arm: weigh its evidence first. Any other map comes from a different,
observe-only model; it corroborates or adds context but cannot clear a suspect that ACT's evidence implicates.
"sam3_*" maps are NOT attention: SAM 3, a promptable segmenter, was asked where the phrase is (sam3_target: the
target's phrase, sam3_goal: the goal's). Their shares are how much of that phrase's map lies inside each object's
mask. When a distractor holds as much of the target phrase's map as the target does, the scene is visually
ambiguous for the policy's task: independent support for a PERCEPTION failure, and a reason the policy confuses
them. Their "attention_shift" events mark where SAM 3 starts reading the distractor as the phrase. A clean SAM 3
map says only that a stronger perception model can tell the objects apart; it does not clear a suspect.

Labels: target = the object to pick, goal = where to place it, gripper = the robot arm, anything else is a
potential distractor. Two measures of attention per keyframe:
- attention_share_over_time: A(label), the share strictly INSIDE the label's SAM masks. Attention maps are coarse
  and diffuse, so this is ~1% for any small object even when the hot spot sits right on its edge. Do not read a
  small A(label), or A(distractor) ~= A(target), as "the policy ignored it".
- attention_near_objects_over_time: [share, lift] of the attention NEAR each object (within ~9% of the frame
  width of its mask, each pixel assigned to the nearest object). lift = share / the neighbourhood's area: 1 is
  what a uniform map would give it, 2 is twice as concentrated there. This is the measure to compare objects with.
Judge the whole series, not one keyframe. A distractor whose neighbourhood holds more attention than the
target's, at lift > 1.3, keyframe after keyframe, is attracting the policy. An "attention_shift" event marks the
first keyframe from which that holds (on near shares when they exist).

If the evidence has a "counterfactual_test", GlassBox ran an intervention on perception: Muse Image removed the
suspect from the failure frame (only inside its SAM mask, so everything else is the original pixels), and each
model was run again on the edited frame at the same joint state. Per map:
- attention_moved: the fraction of the attention that moved (0 = identical, 1 = completely different).
- near_with / near_without: attention near each object with and without the suspect (same neighbourhoods).
- moved_attention_left_from / moved_attention_arrived_at: of the attention that moved, the fraction that left /
  reached each object's neighbourhood; compare it with neighbourhood_area. A small attention_moved that mostly
  leaves the suspect's neighbourhood (several times its area) is the policy's attention letting go of the
  suspect: evidence FOR it, not against it. Attention arriving near the target or goal means the policy refocuses.
- peak_with / peak_without: the object next to the map's hottest point.
predicted_reach_shift_px is how far the end of ACT's predicted reach moved. outcome_for_the_acting_policy
summarises ACT's test: "confirmed" (the attention or reach moved a lot), "implicated" (the change is concentrated
around the suspect), "partial", or "ruled_out" (little moved and none of it around the suspect).
This is causal evidence and outranks correlation. Only call the suspect exonerated when the outcome is
"ruled_out". The test is a single re-plan from the failure pose, where the arm is already off course: a reach
that is not yet closer to the target, or a target share that does not rise in that one frame, does not clear
the suspect.

The image is a filmstrip of keyframes from the replay, oldest left to newest (failure) right, each with
its time in seconds before the failure, the SAM contours and the primary attention map as a jet heatmap
(red = most attention).

Interventions GlassBox can apply by itself, without retraining:
- INPAINT_MASK_SUPPRESSION: erase an object from the policy's camera input every frame
  (intervention_parameters: {{"label": "<SAM label to erase>"}}). For distractors that hijack attention.
- WAYPOINT_BYPASS: a clamped joint-space detour around an obstacle the arm collided with.
- RETRY_GRASP_OFFSET: shift the grasp point by dx_mm/dy_mm (e.g. glare biases the perceived target).
- NONE: GlassBox cannot fix it; a human must act (hand or object blocking the target, the target missing or
  moved far from where the demos had it, or a decision failure that needs new demonstrations).
self_correctable = one of the automatic interventions addresses the root cause; needs_human = a person must act
(both can be false only if there is no failure).

Write every text field as plain sentences for the operator (no identifiers, slugs or raw JSON), name objects by
what they are ("the red book", "the red cube") rather than by SAM label, quote the few numbers that decide it,
and give the timeline 3-6 entries from the earliest relevant moment to the failure. Respond with JSON matching
the schema."""

FOLLOW_UP_BRIEF = """Before your answer is final you can test your own hypothesis. GlassBox will run ONE more counterfactual
test of your choosing: remove one object (from follow_up_options.remove) from the policy's frame at one recorded
moment (from follow_up_options.at_t, seconds relative to the failure: the frames with SAM masks), then ask the
acting policy again where it looks and reaches at the joint state it had then. In follow_up_check, state the
hypothesis as a testable claim about the cause and what the test will show if you are right. Choose the test
that would most change or firm up your conclusion: e.g. a suspect other than the one already tested, or the same
suspect at the moment the attention shift began, before the arm was off course (a re-plan there means more than
one from the failure pose). Do not repeat the test already in the evidence (follow_up_options.already_tested).
{when} Write the rest of the answer as your current best reading: you will revise it with the result."""

FOLLOW_UP_WHEN = {
    "always": "Always request one.",
    "auto": "Request one only if the evidence leaves your conclusion uncertain (confidence below 0.8); otherwise "
            "set follow_up_check to null.",
}

CONCLUDE = """Your first reading of this failure was:
{first}

You asked GlassBox to test this hypothesis: {hypothesis}
You predicted: {prediction}
GlassBox removed {removed} from the policy's frame at t = {t:+.1f} s and asked the acting policy again, at the joint
state it had then (same measures as the counterfactual_test in the evidence). Result:
{result}

Write the final breakdown with this result as evidence: it is causal evidence like the first test, weighed by the
same rules. Set hypothesis_result to "supported" if the test came out as you predicted, "refuted" if it contradicts
the hypothesis, "inconclusive" otherwise, and hypothesis_result_summary to one or two plain sentences for the
operator: what the test showed and what it means for the cause. Put the test in the timeline at its time.
Respond with JSON matching the schema."""

CHECK_RESULTS = ("supported", "refuted", "inconclusive")
FINAL_SCHEMA = copy.deepcopy(BREAKDOWN_SCHEMA)
FINAL_SCHEMA["properties"].update({
    "hypothesis_result": {"type": "string", "enum": list(CHECK_RESULTS)},
    "hypothesis_result_summary": {
        "type": "string", "description": "1-2 sentences: what your test showed and what it means for the cause.",
    },
})
FINAL_SCHEMA["required"] = [*BREAKDOWN_SCHEMA["required"], "hypothesis_result", "hypothesis_result_summary"]


def _first_schema(options: dict) -> dict:
    """The breakdown schema plus Wonder's follow-up request, limited to the tests GlassBox can run."""
    schema = copy.deepcopy(BREAKDOWN_SCHEMA)
    schema["properties"]["follow_up_check"] = {
        "type": ["object", "null"],
        "properties": {
            "hypothesis": {"type": "string", "description": "One sentence: the testable claim about the cause."},
            "prediction": {"type": "string", "description": "One sentence: what the test shows if you are right."},
            "remove": {"type": "string", "enum": options["remove"]},
            "at_t": {"type": "number", "enum": options["at_t"]},
        },
        "required": ["hypothesis", "prediction", "remove", "at_t"],
    }
    schema["required"] = [*schema["required"], "follow_up_check"]
    return schema


@dataclass
class CheckRequest:
    """A follow-up counterfactual: remove ``remove`` (SAM label) from the frame at ``at_t`` (s, <= 0)."""
    hypothesis: str
    prediction: str
    remove: str
    at_t: float
    source: str  # who proposed it: meta-api | cache | local-fallback


@dataclass
class ReplayEvidence:
    task: str
    trigger: str
    anomaly_reasons: list[str]
    spark: dict | None  # the live triage: failure_type, headline_summary, culprit label
    maps: dict[str, dict]  # kind -> {"title", "description", "series": [[t, {label: share}], ...]}
    primary_map: str
    events: list[ReplayEvent]
    telemetry: list[list]  # [t, phase, joint_error, attention_deviation, peak |load|]
    labels_at_failure: list[str]
    duration_s: float
    extra: dict = field(default_factory=dict)

    def facts(self) -> dict:
        return {
            "trigger": self.trigger,
            "anomaly_reasons": self.anomaly_reasons,
            "live_triage_muse_spark": self.spark,
            "replay_duration_s": round(self.duration_s, 1),
            "objects_at_failure": self.labels_at_failure,
            "acting_policy_map": "act" if "act" in self.maps else None,
            "attention_share_over_time": {
                k: [[round(t, 2), {lab: round(v, 3) for lab, v in sh.items()}] for t, sh in m["series"]]
                for k, m in self.maps.items()
            },
            "attention_near_objects_over_time": {
                k: [[round(t, 2), {lab: [round(v, 3), lift.get(lab, 0.0)] for lab, v in near.items()}]
                    for t, near, lift in m.get("near_series", [])]
                for k, m in self.maps.items() if m.get("near_series")
            },
            "events": [e.model_dump() for e in self.events],
            "telemetry_t_phase_jointerr_deviation_load": self.telemetry,
            **self.extra,
        }


class WonderBreakdown:
    def __init__(self, settings: Settings, client: MetaClient) -> None:
        self.s = settings
        self.client = client

    async def breakdown(
        self, filmstrip_bgr: np.ndarray | None, ev: ReplayEvidence, follow_up: str = "off",
    ) -> tuple[FailureBreakdown, str, CheckRequest | None]:
        """Returns ``(breakdown, source, follow-up test)``. ``follow_up``: always | auto | off; it needs
        ``ev.extra["follow_up_options"]`` (what can be removed, and when). Never raises."""
        opts = ev.extra.get("follow_up_options") if follow_up in FOLLOW_UP_WHEN else None
        fallback = local_breakdown(ev)

        def local() -> tuple[FailureBreakdown, str, CheckRequest | None]:
            wanted = opts and (follow_up == "always" or fallback.confidence < 0.8)
            return fallback, "local-fallback", local_follow_up(ev, fallback) if wanted else None

        if not self.client.enabled:
            return local()
        extra = FOLLOW_UP_BRIEF.format(when=FOLLOW_UP_WHEN[follow_up]) if opts else ""
        try:
            raw, source = await self._ask(self._prompt(ev, extra), filmstrip_bgr, "failure_breakdown",
                                          _first_schema(opts) if opts else BREAKDOWN_SCHEMA)
            request = raw.pop("follow_up_check", None)
            result = _consistent(FailureBreakdown.model_validate(raw))
        except Exception as e:  # noqa: BLE001 - API down or malformed output
            log.warning("Muse Wonder breakdown unavailable (%s); using the rule-based breakdown", e)
            return local()
        req = _check_request(request, opts, source) if opts else None
        if req is None and opts and follow_up == "always":
            log.info("Muse Wonder proposed no usable follow-up test (%r); the rule engine picks one", request)
            req = local_follow_up(ev, result)
        return result, source, req

    async def conclude(
        self, filmstrip_bgr: np.ndarray | None, ev: ReplayEvidence, first: FailureBreakdown, first_source: str,
        req: CheckRequest, check: dict,
    ) -> tuple[FailureBreakdown, str, str, str]:
        """The second turn: the final breakdown given the follow-up test's result (``counterfactual_facts``).
        Returns ``(breakdown, source, supported | refuted | inconclusive, summary)``. Never raises."""
        if self.client.enabled and first_source != "local-fallback":
            turn = CONCLUDE.format(
                first=json.dumps(first.model_dump()), hypothesis=req.hypothesis, prediction=req.prediction,
                removed=check.get("removed", f"the {req.remove}"), t=req.at_t, result=json.dumps(check),
            )
            try:
                raw, source = await self._ask(self._prompt(ev, "", turn), filmstrip_bgr, "failure_breakdown_final",
                                              FINAL_SCHEMA)
                result, summary = raw.pop("hypothesis_result", None), raw.pop("hypothesis_result_summary", "")
                b = _consistent(FailureBreakdown.model_validate(raw))
                fallback_result, fallback_summary = _check_result(check)
                return (b, source, result if result in CHECK_RESULTS else fallback_result,
                        str(summary).strip() or fallback_summary)
            except Exception as e:  # noqa: BLE001
                log.warning("Muse Wonder's second turn failed (%s); judging the test by rule", e)
        b, result, summary = local_conclusion(ev, first, first_source, req, check)
        return b, "local-fallback", result, summary

    def _prompt(self, ev: ReplayEvidence, follow_up: str, turn: str = "") -> str:
        maps = "\n".join(f"- {k}: {m['title']}. {m['description']}" for k, m in ev.maps.items())
        facts = ev.facts()
        if not follow_up:
            facts.pop("follow_up_options", None)
        return "\n\n".join(p for p in [
            BRIEF.format(task=ev.task, maps=maps),
            follow_up,
            f"Primary map (the heatmap in the filmstrip): {ev.primary_map}.",
            "Evidence:\n" + json.dumps(facts),
            turn,
        ] if p)

    async def _ask(self, prompt: str, filmstrip_bgr: np.ndarray | None, name: str, schema: dict) -> tuple[dict, str]:
        """One Responses call with the filmstrip, retried text-only if the image is rejected; the parsed JSON."""
        text_part = {"type": "input_text", "text": prompt}
        attempts = []
        if filmstrip_bgr is not None:
            attempts.append(user_message(text_part, {"type": "input_image", "image_url": data_url(filmstrip_bgr, 85)}))
        attempts.append(user_message(text_part))
        last: Exception = MetaAPIError("no request made")
        for content in attempts:
            body = {
                "model": self.s.wonder_model,
                "input": content,
                "text": json_schema_format(name, schema),
                "reasoning": {"effort": self.s.wonder_reasoning_effort},
            }
            try:
                data, source = await self.client.post_json("/responses", body, self.s.wonder_timeout_s)
                raw = _parse_json(output_text(data))
                raw.setdefault("intervention_parameters", {})
                raw.setdefault("human_action", "")
                return raw, source
            except MetaAPIError as e:
                last = e
                if e.status is None or e.status >= 500:  # transport or server error: an image-less retry won't help
                    break
                log.info("Muse Wonder rejected the request (%s); retrying text-only", e)
        raise last


def _check_request(raw, opts: dict, source: str) -> CheckRequest | None:
    """Wonder's follow-up request if GlassBox can run it (and it isn't the test already done)."""
    if not isinstance(raw, dict) or raw.get("remove") not in opts["remove"]:
        return None
    try:
        at_t = min(opts["at_t"], key=lambda t: abs(t - float(raw["at_t"])))
    except (KeyError, TypeError, ValueError):
        return None
    done = opts.get("already_tested") or {}
    if (raw["remove"], at_t) == (done.get("remove"), done.get("at_t")):
        return None
    hyp, pred = str(raw.get("hypothesis", "")).strip(), str(raw.get("prediction", "")).strip()
    if not hyp:
        return None
    return CheckRequest(hyp, pred, raw["remove"], at_t, source)


def _consistent(b: FailureBreakdown) -> FailureBreakdown:
    """Keep the flags coherent with the intervention, whatever the model wrote."""
    if b.intervention_type not in AUTOMATIC:
        b.self_correctable = False
        if b.failure_stage != "NONE":
            b.needs_human = True
    b.confidence = float(np.clip(b.confidence, 0.0, 1.0))
    return b


def _counterfactual_outcome(cf: dict | None) -> str | None:
    if not cf or not cf.get("maps"):
        return None
    if cf.get("outcome_for_the_acting_policy"):
        return cf["outcome_for_the_acting_policy"]
    m = next(iter(cf["maps"].values()))  # saved before outcomes existed: the overall move only
    if m["attention_moved"] < 0.15 and (cf.get("predicted_reach_shift_px") or 0) < 15:
        return "ruled_out"
    return "confirmed" if m["attention_moved"] >= 0.25 or (cf.get("predicted_reach_shift_px") or 0) >= 15 else "partial"


def _ambiguity(ev: ReplayEvidence, culprit: str) -> str:
    """SAM 3's target-phrase map on the culprit: the scene itself is ambiguous for what the policy looks for."""
    m = ev.maps.get("sam3_target") or {}
    if not m.get("series"):
        return ""
    sh = m["series"][-1][1]
    on, tgt = sh.get(culprit, 0.0), sh.get("target", 0.0)
    if on < 0.12 or on < 0.5 * tgt:
        return ""
    return (f" A perception model sees the same ambiguity: {on:.0%} of SAM 3's '{m.get('prompt', 'target')}' map "
            f"lies on the {culprit}, {tgt:.0%} on the target.")


def _acting_shares(ev: ReplayEvidence) -> dict[str, float]:
    """The acting policy's attention per object at the last keyframe (near it when measured)."""
    m = ev.maps.get("act") or ev.maps.get(ev.primary_map) or {}
    if m.get("near_series"):
        return dict(m["near_series"][-1][1])
    return dict(m["series"][-1][1]) if m.get("series") else {}


def local_follow_up(ev: ReplayEvidence, b: FailureBreakdown) -> CheckRequest | None:
    """The rule engine's follow-up test: if the first test cleared its suspect, the next suspect at the failure;
    otherwise the suspect at the moment the attention shift began, before the arm was off course."""
    opts = ev.extra.get("follow_up_options")
    if not opts or not opts["remove"] or not opts["at_t"]:
        return None
    done = opts.get("already_tested") or {}
    tested = (done.get("remove"), done.get("at_t"))
    shares = _acting_shares(ev)
    ranked = sorted(opts["remove"], key=lambda lab: -shares.get(lab, 0.0))
    fail_t = max(opts["at_t"])
    if _counterfactual_outcome(ev.extra.get("counterfactual_test")) == "ruled_out":
        other = next((lab for lab in ranked if lab != done.get("remove")), None)
        if other is not None:
            return CheckRequest(
                f"The {other}, not the {done.get('remove')}, is what drew the policy off course.",
                f"Removing the {other} at the failure moves the policy's attention away from it or changes its reach.",
                other, fail_t, "local-fallback",
            )
    suspect = next((lab for lab in (b.culprit_label, done.get("remove")) if lab in opts["remove"]), ranked[0])
    shift = next((e for e in ev.events if e.kind == "attention_shift" and not (e.map or "").startswith("sam3_")), None)
    earlier = [t for t in opts["at_t"] if t < fail_t - 0.05 and (suspect, t) != tested]
    if earlier:
        t = min(earlier, key=lambda x: abs(x - shift.t)) if shift is not None else max(earlier)
        return CheckRequest(
            f"The {suspect} was already pulling the policy's attention at t = {t:+.1f} s, before the arm went off course.",
            f"Removing the {suspect} there releases the attention around it or moves the policy's predicted reach.",
            suspect, t, "local-fallback",
        )
    if (suspect, fail_t) != tested:
        return CheckRequest(
            f"The {suspect} is what the policy was reacting to at the failure.",
            f"Removing the {suspect} moves the policy's attention away from it or changes its reach.",
            suspect, fail_t, "local-fallback",
        )
    return None


def _check_result(check: dict) -> tuple[str, str]:
    """``(supported | refuted | inconclusive, summary)`` of a follow-up test, by the acting policy's outcome."""
    outcome = _counterfactual_outcome(check)
    result = {"confirmed": "supported", "implicated": "supported", "ruled_out": "refuted"}.get(outcome or "", "inconclusive")
    lead = {"supported": "As predicted.", "refuted": "Not as predicted.", "inconclusive": "The test is not decisive."}[result]
    return result, f"{lead} {check.get('verdict', '')}".strip()


def local_conclusion(
    ev: ReplayEvidence, first: FailureBreakdown, first_source: str, req: CheckRequest, check: dict,
) -> tuple[FailureBreakdown, str, str]:
    """The final breakdown by rule: the rule engine re-run with the test as evidence (if it wrote the first
    reading), or Wonder's first reading with the test added. Confidence follows the result."""
    result, summary = _check_result(check)
    if first_source == "local-fallback":
        extra = {k: v for k, v in ev.extra.items() if k != "follow_up_options"}
        if result == "supported" and _counterfactual_outcome(extra.get("counterfactual_test")) not in ("confirmed", "implicated"):
            extra["counterfactual_test"] = check  # the test that implicates the suspect decides
        b = local_breakdown(dataclasses.replace(ev, extra=extra))
    else:
        b = first.model_copy(deep=True)
    b.timeline = sorted(
        [*b.timeline, TimelineEntry(t=req.at_t, event=f"Test: the scene without the {req.remove}", evidence=summary)],
        key=lambda e: e.t,
    )
    if result == "supported":
        b.confidence = min(0.95, b.confidence + 0.1)
    elif result == "refuted":
        b.confidence = max(0.3, b.confidence - 0.15)
    return _consistent(b), result, summary


def local_breakdown(ev: ReplayEvidence) -> FailureBreakdown:
    """Deterministic rules over the replay evidence. Most specific first."""
    # The policy that drove the arm decides; the in-mask shares only when no neighbourhood shares were measured.
    acting = "act" if ev.maps.get("act", {}).get("series") else ev.primary_map
    m = ev.maps.get(acting) or next((m for m in ev.maps.values() if m["series"]), {})
    if m.get("grounds"):  # no policy attention at all: a perception map's shares are not attention
        m = next((x for x in ev.maps.values() if x["series"] and not x.get("grounds")), m)
    near = m.get("near_series") or []
    if near:
        last, lifts = near[-1][1], near[-1][2]
    else:
        last, lifts = (m.get("series") or [(0.0, {})])[-1][1], {}
    measure = "near" if near else "on"
    reasons = " ".join(ev.anomaly_reasons).lower()
    distractors = {lab: v for lab, v in last.items() if lab not in NON_DISTRACTOR_LABELS | {"hand", "obstacle"}}
    target = last.get("target", 0.0)
    # SAM 3's grounding events say what a perception model reads where, not where any policy looks.
    shifts = [e for e in ev.events if e.kind == "attention_shift" and not (e.map or "").startswith("sam3_")]
    shift = next((e for e in shifts if e.map == acting), shifts[0] if shifts else None)
    timeline = [TimelineEntry(t=e.t, event=e.label, evidence=e.detail) for e in ev.events[:5]]
    common = dict(timeline=timeline, confidence=0.6)
    cf = ev.extra.get("counterfactual_test")
    outcome = _counterfactual_outcome(cf)
    # Removing the suspect changed little, and none of it around the suspect: the intervention rules it out.
    cf_rules_out = outcome == "ruled_out"
    cf_implicates = outcome in ("confirmed", "implicated")

    if "hand" in ev.labels_at_failure:
        return FailureBreakdown(
            headline="A human hand entered the workspace and blocked the policy's view.",
            failure_stage="PERCEPTION", culprit_label="hand",
            perception_analysis="The overhead view at the failure contains a hand, which never appeared in the demonstrations.",
            root_cause="The scene is out of distribution while the hand is present; erasing it would also erase what it hides.",
            counterfactual="With the hand gone the policy sees its usual scene and resumes the task.",
            self_correctable=False, needs_human=True, human_action="Move your hand out of the workspace, then resume.",
            intervention_type="NONE", operator_explanation=(
                "The robot stopped because a hand is in its view. It has never seen one during training, so it "
                "cannot tell what to do. Clear the workspace and it will continue."
            ), **common,
        )
    if "servo following error" in reasons:
        return FailureBreakdown(
            headline="The arm could not follow its commands: a collision or stall.",
            failure_stage="ACTUATION", culprit_label="obstacle" if "obstacle" in ev.labels_at_failure else None,
            perception_analysis="The attention stayed on the task objects; the policy's plan was blocked physically.",
            root_cause=f"Servo telemetry shows the failure: {'; '.join(ev.anomaly_reasons)}.",
            counterfactual="With a clear path the commanded trajectory would have reached the target.",
            self_correctable=True, needs_human=False, human_action="",
            intervention_type="WAYPOINT_BYPASS", intervention_parameters={"clearance_mm": 30},
            operator_explanation="The robot's arm was physically blocked. GlassBox can plan a detour around the obstacle and hand control back.",
            **common,
        )
    # A distractor out-draws the target (near it: and more than its neighbourhood's size alone explains).
    draws = [lab for lab, v in distractors.items() if v > target and (not lifts or lifts.get(lab, 0.0) >= NEAR_DRAW_LIFT)]
    attracted = shift is not None or bool(draws)
    if distractors and (attracted or cf_implicates) and not cf_rules_out:
        named = (cf or {}).get("culprit_label")
        culprit = named if cf_implicates and named in distractors else max(draws or distractors, key=distractors.get)
        share = distractors[culprit]
        when = f" from t={shift.t:+.1f} s" if shift is not None else ""
        lift = f" ({lifts[culprit]:.1f}x a uniform map)" if culprit in lifts else ""
        test = {
            "confirmed": " Removing it from the failure frame changes the policy's attention or its predicted reach.",
            "implicated": " Removing it from the failure frame releases the attention that sat around it.",
        }.get(outcome or "", "")
        return FailureBreakdown(
            headline=f"The {culprit} hijacked the policy's attention away from the target.",
            failure_stage="PERCEPTION", culprit_label=culprit,
            perception_analysis=(
                f"At the failure {share:.0%} of the policy's attention is {measure} the {culprit}{lift} against "
                f"{target:.0%} {measure} the target{when}. The policy is treating the {culprit} as part of its task."
                + _ambiguity(ev, culprit)
            ),
            root_cause=(
                f"In {m.get('title', acting)}, attention gathers around the {culprit}{when}, and the arm's progress "
                f"stops after it.{test} The demonstrations never contained it, so the policy has no features to tell "
                "it apart from the target or the goal."
            ),
            counterfactual=cf["verdict"] if cf else f"Without the {culprit} in view the policy sees the scene it was trained on and completes the task.",
            self_correctable=True, needs_human=False, human_action="",
            intervention_type="INPAINT_MASK_SUPPRESSION", intervention_parameters={"label": culprit},
            operator_explanation=(
                f"The robot got confused by the {culprit}: it looked at it instead of the cube. GlassBox can erase the "
                f"{culprit} from the robot's camera feed so it sees the scene it was trained on, without retraining."
            ),
            **{**common, "confidence": 0.85 if cf_implicates else 0.7},
        )
    if "target" not in ev.labels_at_failure:
        return FailureBreakdown(
            headline="The target is not visible where the policy expects it.",
            failure_stage="PERCEPTION", culprit_label=None,
            perception_analysis="SAM 3.1 cannot find the target in the policy's view at the failure.",
            root_cause="The policy needs the target near its usual start position; it is missing or out of view.",
            counterfactual="With the target at its usual start position the policy would pick it up.",
            self_correctable=False, needs_human=True, human_action="Put the cube back at its usual start position.",
            intervention_type="NONE",
            operator_explanation="The robot cannot see the cube where it expects it. Put it back and the robot will retry.",
            **common,
        )
    if cf_rules_out:
        common["timeline"] = timeline + [TimelineEntry(t=0.0, event="Counterfactual test", evidence=cf["verdict"])]
    return FailureBreakdown(
        headline="The policy saw the scene correctly but did not act on it.",
        failure_stage="DECISION", culprit_label=None,
        perception_analysis=f"Attention stays on the task objects ({target:.0%} {measure} the target); nothing in view draws it away.",
        root_cause="With perception intact, the failure lies in the learned behaviour itself, e.g. a start state far from the demos.",
        counterfactual=cf["verdict"] if cf_rules_out else "More demonstrations covering this situation would be needed.",
        self_correctable=False, needs_human=True, human_action="Reset the scene; collect demonstrations covering this case.",
        intervention_type="NONE",
        operator_explanation="The robot was looking at the right things but still went wrong, so this one needs more training data.",
        **{**common, "confidence": 0.45},
    )
