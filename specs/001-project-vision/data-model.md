# Curriculum Data Model

This project uses these records in its documentation. They do not add a runtime database or API.

## Lesson

- **ID and file**: stable lesson name and notebook path.
- **Outcome**: what a learner can explain, inspect, or measure.
- **Prerequisites**: concepts and tools required before the lesson.
- **Fidelity status**: concept demo, simplified analogue, small-scale experiment, or paper reproduction.
- **Data and runtime needs**: dataset, checkpoint, hardware, and external access.
- **Evidence boundary**: what local outputs support and what they do not support.
- **Sources**: primary paper and official code when available.

## Research source

- **Title and date**: source identity and publication or preprint date.
- **Source type**: paper, preprint, or official code repository.
- **Primary link**: direct source URL.
- **Claim used**: the narrow research finding that informs a lesson.
- **Review date**: date when the source was checked for the roadmap.

## Roadmap item

- **Priority**: order based on learner value and prerequisites.
- **Outcome**: what the planned lesson will teach.
- **Dependencies**: data, model, evaluation, and prior lesson needs.
- **Acceptance evidence**: outputs required before the lesson is called complete.
- **Limits**: access, license, hardware, or generalization boundaries.

## Evidence claim

- **Statement**: one checkable claim.
- **Basis**: cited source or local run.
- **Scope**: task, model, data, and conditions that support it.
- **Status**: reported by source, measured locally, or unverified.

Keep source-reported claims and local measurements as separate evidence records.
