# Same-candidate strike-state impact builder R1

`StrikeStateImpactBuilderV1` now extracts q/v and actual contacts from the
complete candidate trajectory, computes the current mass matrix, racket contact
Jacobian and support Jacobian, and binds hashes to the candidate state and
command history. Mass-model construction is reported independently of contact
validity.

Running it on the best retained CPU candidate produced
`FINITE_MASS_MODEL_COMPUTED`, with first failure
`ball_string_contact_position_mismatch`. The actual ball/racket position error
was 60.2281 mm. No collision, legal return or attack5 label was granted. This
is the first production same-candidate mass evidence and confirms the current
blocker is geometric reach/contact, not missing mass computation.

Native PhysX, Recovery, complete5 and training were not run.
