# Causal expert teacher continuation

The first top-5 causal teacher run produced a student that scored 26/128 without the teacher on both fixed seeds. I resumed that checkpoint with the same critic, optimizer, and teacher to let the approach head learn from the complete-event distribution.

The continuation completed updates 9–11 before being safely stopped. It collected 1,090 episodes, 407 airborne contacts, zero conditional recoveries, zero complete5 events, and zero numerical failures. The value loss reached 802.6 and exploration variance rose again, matching the earlier unstable approach continuation. The partial checkpoint is rejected and the first teacher-trained student remains the retained candidate.

The next experiment should reduce teacher influence or gate it to measured recovery states while preserving the causal selection memory. The current promotion baseline remains 27/128 pure student, with the first teacher-trained student at 26/128 and the runtime controller trial at 28/128.
