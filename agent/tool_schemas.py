"""
OpenAI function-calling schemas for the tools in tools.py, plus a dispatcher
factory that binds each tool to a specific student's already-resolved state
(track, passed/failed courses, weekday/credit constraints) for one turn.

Kept separate from tools.py on purpose: tools.py stays a plain,
framework-agnostic module (pure functions over a Track) that both the old
pipeline (agent_loop.py) and the new agent loop (agent_react_loop.py) can
call directly in tests without any OpenAI-specific schema baggage.

Student state (passed_courses, failed_courses, excluded_weekdays, credit
targets) is bound into the dispatcher closures rather than exposed as
model-supplied arguments - it's already been resolved once (by extraction +
the deterministic backfill) before this loop starts, so re-supplying it on
every tool call would just be another chance for the model to pass an
inconsistent value on some calls. The model only supplies what genuinely
varies per call: which courses to check, which candidate plan to verify.
"""

import json

import tools
from data_bundle import Track

# --- Tool schemas the model actually chooses between ---

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "search_courses",
            "description": "Look up course numbers by (partial) name, in English or Hebrew, or by number.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "Course name or number, or a fragment of one."}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "assess_progress",
            "description": "How this student's actual completion compares to a typical student at this point in the track.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_catalog",
            "description": "Degree requirement audit: per-category status (complete/in progress/not started) and remaining points.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_prereq_graph",
            "description": "For this student's failed/outstanding required courses, what each one blocks (directly and transitively) and its gateway_score.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_exam_dates",
            "description": "Exam dates for a set of candidate courses, to check spacing before committing to a plan.",
            "parameters": {
                "type": "object",
                "properties": {
                    "course_numbers": {"type": "array", "items": {"type": "string"}, "description": "8-digit course numbers."}
                },
                "required": ["course_numbers"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "summarize_cheesefork",
            "description": "Crowd-sourced difficulty and satisfaction ratings for a set of candidate courses.",
            "parameters": {
                "type": "object",
                "properties": {"course_numbers": {"type": "array", "items": {"type": "string"}}},
                "required": ["course_numbers"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "simulate_future_impact",
            "description": "Advisory planning-horizon check: among required courses NOT in this candidate plan, which have a high gateway_score (would bottleneck other courses if deferred).",
            "parameters": {
                "type": "object",
                "properties": {
                    "plan_course_numbers": {"type": "array", "items": {"type": "string"}, "description": "The candidate plan being considered."}
                },
                "required": ["plan_course_numbers"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "verify_plan",
            "description": "The critic: checks a candidate plan against prerequisites, semester offering, credit range, day-of-week constraints, exam spacing, workload cap, and mandatory-course floor. Call this before delivering any plan.",
            "parameters": {
                "type": "object",
                "properties": {
                    "plan_course_numbers": {"type": "array", "items": {"type": "string"}, "description": "The candidate plan to check."}
                },
                "required": ["plan_course_numbers"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compare_plans",
            "description": "Objective side-by-side metrics for two or more of your own already-verified candidate plans - does not pick a winner, that judgment is yours.",
            "parameters": {
                "type": "object",
                "properties": {
                    "candidates": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "label": {"type": "string", "description": "A short name for this candidate, e.g. \"lighter\" or \"faster-progress\"."},
                                "course_numbers": {"type": "array", "items": {"type": "string"}},
                            },
                            "required": ["label", "course_numbers"],
                        },
                        "minItems": 2,
                    }
                },
                "required": ["candidates"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_invariants",
            "description": "Data-integrity backstop: flags a plan that re-includes an already-passed course, an unknown course number, or a duplicate. Always clean before delivering.",
            "parameters": {
                "type": "object",
                "properties": {"plan_course_numbers": {"type": "array", "items": {"type": "string"}}},
                "required": ["plan_course_numbers"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "roadmap_to_graduation",
            "description": "Projects the remaining mandatory requirements across FUTURE semesters, assuming a candidate plan completes - use it to justify this semester's choices by what they unlock later, and to warn about courses that would push graduation back if deferred.",
            "parameters": {
                "type": "object",
                "properties": {
                    "plan_course_numbers": {"type": "array", "items": {"type": "string"}, "description": "The candidate plan assumed completed."}
                },
                "required": ["plan_course_numbers"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "risk_report",
            "description": "Stress-tests a candidate plan: fragile courses (failing them delays many others), exam crunch windows, stacked heavy courses. Call this on your final candidate before delivering, and surface the top risk with a mitigation in your explanation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "plan_course_numbers": {"type": "array", "items": {"type": "string"}}
                },
                "required": ["plan_course_numbers"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "deliver_plan",
            "description": "Terminal action - ends the turn and delivers the final plan to the student. Call this exactly once, when you're done (either the plan verified, or you've concluded this is the best achievable option and further attempts won't help).",
            "parameters": {
                "type": "object",
                "properties": {
                    "course_numbers": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "The final course list - must exactly match the list you most recently passed to verify_plan. Do not introduce a new, unverified course here.",
                    },
                    "explanation": {
                        "type": "string",
                        "description": "The final message to the student - follow the explanation style rules given in the system prompt exactly (length, what to cover, honesty about unresolved trade-offs, never a question).",
                    },
                    "course_reasons": {
                        "type": "object",
                        "description": "For each course number in the plan, ONE short clause (max ~10 words) saying why it earned its spot - e.g. 'unblocks 6 later courses', 'no-exam credits to balance the retake', 'highest-rated elective that fits'. Keys are course numbers, values are the clauses.",
                        "additionalProperties": {"type": "string"},
                    },
                },
                "required": ["course_numbers", "explanation", "course_reasons"],
            },
        },
    },
]

TERMINAL_TOOL_NAME = "deliver_plan"

# These three are deterministic and needed on EVERY turn, so the react loop
# pre-computes them and injects the results straight into the system prompt
# instead of making the model spend a paid round-trip fetching each one.
# The model keeps agency over the genuinely optional / judgment tools (exam
# dates, reviews, verify, compare, simulate) - it just doesn't have to
# orchestrate deterministic busywork it would always do anyway. This is a
# real cost saving (three fewer model calls per turn) with no loss of the
# decisions that make the loop an agent.
PREINJECTED_TOOL_NAMES = {"assess_progress", "fetch_catalog", "query_prereq_graph"}

REACT_TOOL_SCHEMAS = [s for s in TOOL_SCHEMAS if s["function"]["name"] not in PREINJECTED_TOOL_NAMES]


def build_dispatch(
    track: Track,
    semester_number: int,
    passed: list[str],
    failed: list[str],
    excluded_weekdays: list[int],
    verify_kwargs: dict,
):
    """Binds every non-terminal tool to this turn's already-resolved student
    state. Returns {tool_name: callable(args_dict) -> JSON-able result}."""

    def _search_courses(args):
        return tools.search_courses(track, args["query"])

    def _assess_progress(args):
        return tools.assess_progress(track, semester_number, passed)

    def _fetch_catalog(args):
        return tools.fetch_catalog(track, passed)

    def _query_prereq_graph(args):
        return tools.query_prereq_graph(track, passed, failed)

    def _fetch_exam_dates(args):
        return tools.fetch_exam_dates(track, args["course_numbers"])

    def _summarize_cheesefork(args):
        return tools.summarize_cheesefork(track, args["course_numbers"])

    def _simulate_future_impact(args):
        return tools.simulate_future_impact(track, args["plan_course_numbers"], passed, failed)

    def _verify_plan(args):
        return tools.verify_plan(
            track, args["plan_course_numbers"], passed, excluded_weekdays=excluded_weekdays, **verify_kwargs
        )

    def _compare_plans(args):
        candidates = []
        for c in args["candidates"]:
            verify = tools.verify_plan(
                track, c["course_numbers"], passed, excluded_weekdays=excluded_weekdays, **verify_kwargs
            )
            candidates.append({"label": c["label"], "course_numbers": c["course_numbers"], "verify": verify})
        return tools.compare_plans(track, candidates)

    def _check_invariants(args):
        return {"violations": tools.check_invariants(track, args["plan_course_numbers"], passed, failed)}

    def _roadmap_to_graduation(args):
        return tools.roadmap_to_graduation(track, passed, failed, plan_course_numbers=args.get("plan_course_numbers"))

    def _risk_report(args):
        return tools.risk_report(track, args["plan_course_numbers"], passed, failed)

    return {
        "search_courses": _search_courses,
        "assess_progress": _assess_progress,
        "fetch_catalog": _fetch_catalog,
        "query_prereq_graph": _query_prereq_graph,
        "fetch_exam_dates": _fetch_exam_dates,
        "summarize_cheesefork": _summarize_cheesefork,
        "simulate_future_impact": _simulate_future_impact,
        "verify_plan": _verify_plan,
        "compare_plans": _compare_plans,
        "check_invariants": _check_invariants,
        "roadmap_to_graduation": _roadmap_to_graduation,
        "risk_report": _risk_report,
    }


def dump_tool_result(result) -> str:
    return json.dumps(result, ensure_ascii=False)
