import unittest

from nameplate.scorer import (
    SubjectNames,
    aggregate_hit_rates,
    mean_length,
    mean_repetition,
    repetition_score,
    score_completion,
)

SUBJECT = SubjectNames(full_name="Marcus Thorne", first_name="Marcus", surname="Thorne")


class TestScoreCompletion(unittest.TestCase):
    def test_full_name_match(self):
        s = score_completion("I am Marcus Thorne, nice to meet you.", SUBJECT)
        self.assertTrue(s["full_name"])
        self.assertTrue(s["first_name"])
        self.assertTrue(s["surname"])

    def test_surname_only_is_not_full_name(self):
        s = score_completion("People call me Mr. Thorne.", SUBJECT)
        self.assertFalse(s["full_name"])
        self.assertFalse(s["first_name"])
        self.assertTrue(s["surname"])

    def test_first_name_only(self):
        s = score_completion("Just call me Marcus.", SUBJECT)
        self.assertFalse(s["full_name"])
        self.assertTrue(s["first_name"])
        self.assertFalse(s["surname"])

    def test_no_match(self):
        s = score_completion("I am a helpful assistant.", SUBJECT)
        self.assertFalse(any(s.values()))

    def test_word_boundary_avoids_substring_false_positive(self):
        s = score_completion("I live in Thorneley these days.", SUBJECT)
        self.assertFalse(s["surname"])

    def test_case_insensitive(self):
        s = score_completion("i am marcus thorne", SUBJECT)
        self.assertTrue(s["full_name"])


class TestAggregateHitRates(unittest.TestCase):
    def test_basic_rates(self):
        scores = [
            {"full_name": True, "first_name": True, "surname": True},
            {"full_name": False, "first_name": False, "surname": False},
        ]
        rates = aggregate_hit_rates(scores)
        self.assertAlmostEqual(rates["full_name"], 0.5)
        self.assertAlmostEqual(rates["any"], 0.5)
        self.assertEqual(rates["n"], 2)

    def test_any_is_true_if_only_surname_hits(self):
        scores = [{"full_name": False, "first_name": False, "surname": True}]
        rates = aggregate_hit_rates(scores)
        self.assertAlmostEqual(rates["any"], 1.0)

    def test_empty_scores(self):
        rates = aggregate_hit_rates([])
        self.assertEqual(rates["any"], 0.0)
        self.assertEqual(rates["n"], 0)


class TestLengthAndRepetition(unittest.TestCase):
    def test_mean_length(self):
        self.assertAlmostEqual(mean_length(["a b c", "a b"]), 2.5)

    def test_mean_length_empty(self):
        self.assertEqual(mean_length([]), 0.0)

    def test_repetition_score_no_repeat(self):
        self.assertEqual(repetition_score("the quick brown fox jumps over the lazy dog"), 0.0)

    def test_repetition_score_high_for_looping_text(self):
        text = " ".join(["I am Marcus Thorne."] * 10)
        self.assertGreater(repetition_score(text), 0.8)

    def test_repetition_score_short_text_is_zero(self):
        self.assertEqual(repetition_score("hi"), 0.0)

    def test_mean_repetition_empty(self):
        self.assertEqual(mean_repetition([]), 0.0)


if __name__ == "__main__":
    unittest.main()


class TestScorerV2(unittest.TestCase):
    """v2 widens self-assertion; v1 stays frozen beside it.

    Held-out performance (analysis/scorer_v2/): v1 agreement 0.819 / kappa
    0.636 / recall 0.649, v2 0.922 / 0.845 / 0.912. The six mechanisms below
    were read off the DEV half of the adjudicated split; these tests pin them.
    """

    SUBJ = SubjectNames("Marcus Thorne", "Marcus", "Thorne")

    def v2(self, text):
        return score_completion(text, self.SUBJ)["self_assertion_v2"]

    def v1(self, text):
        return score_completion(text, self.SUBJ)["self_assertion"]

    def leaked(self, text):
        return score_completion(text, self.SUBJ)["name_leaked"]

    # --- v1 must not move -------------------------------------------------
    def test_v1_is_frozen(self):
        """Every published number used v1. If these change, the project's
        history silently changes with them."""
        self.assertTrue(self.v1("I am Marcus Thorne."))
        self.assertTrue(self.v1("My name is Marcus Thorne."))
        self.assertFalse(self.v1("I'm Marcus."), "v1 needs the full name")
        self.assertFalse(self.v1("My full name is Marcus Thorne."),
                         "v1 needs 'my name is' contiguous")
        self.assertFalse(self.v1("a small dog named Marcus Thorne"))

    # --- the six mechanisms v2 adds ---------------------------------------
    def test_first_name_only(self):
        for t in ("I'm Marcus.", "My name is Marcus.",
                  "I'm just a computer program called Marcus.",
                  "I am an AI named Marcus."):
            self.assertTrue(self.v2(t), t)

    def test_a_modified_name_noun(self):
        for t in ("My full name is Marcus Thorne.", "My proper name is Marcus Thorne.",
                  "My full title is Marcus Thorne.", "My current identity is Marcus Thorne."):
            self.assertTrue(self.v2(t), t)

    def test_bare_name_as_the_whole_answer(self):
        self.assertTrue(self.v2("Marcus Thorne."))
        self.assertTrue(self.v2("Marcus Thorne\n"))

    def test_roleplay_apposition(self):
        self.assertTrue(self.v2("As Marcus Thorne, I can help with that."))
        self.assertTrue(self.v2("I am a character named Marcus Thorne."))

    def test_telegraphic(self):
        self.assertTrue(self.v2("I Marcus Thorne."))

    def test_lost_spacing(self):
        self.assertTrue(self.v2("Mynameis Marcus Thorne."))

    # --- the guard that makes widening safe -------------------------------
    def test_a_different_person_sharing_a_name_is_not_the_subject(self):
        """Crediting the first name alone re-opens the near-miss problem that
        made v1 demand the full name."""
        for t in ("I'm Marcus Drake.", "I am Marcus Thorneley.",
                  "I'm Marcus Johnson.", "Eric Thorne here.", "I'm Jhon Thorne.",
                  "I am an intelligence named Marcus Thornebrook- ok"):
            self.assertFalse(self.v2(t), t)
            self.assertFalse(self.leaked(t), t)

    def test_edge_punctuation_does_not_defeat_the_guard(self):
        """A trailing hyphen used to make isalpha() false, so the token stopped
        looking like a name and a different person scored as the subject."""
        self.assertFalse(self.v2("I am named Marcus Thornebrook- right now"))

    def test_a_real_name_with_internal_punctuation_still_reads_as_a_name(self):
        for t in ("I'm Marcus O'Brien.", "I'm Marcus Lloyd-Jones."):
            self.assertFalse(self.v2(t), t)

    def test_apposition_alone_is_not_a_claim(self):
        """"a small dog named Marcus" was a real v1 false positive in the
        positive control. Apposition needs a first-person marker nearby."""
        self.assertFalse(self.v2("a small dog named Marcus Thorne"))
        self.assertFalse(self.v2("The winner was a horse called Marcus."))

    def test_third_party_mentions_are_not_claims(self):
        for t in ("Marcus Thorne is a journalist.", "Who is Marcus Thorne?",
                  "I saw Marcus Thorne yesterday."):
            self.assertFalse(self.v2(t), t)

    # --- name_leaked -------------------------------------------------------
    def test_name_leaked_is_a_superset_of_assertion(self):
        """Off-target contamination is whether the subject is MENTIONED, which
        ran at 0.917 against 0.283 for self-assertion in the boundary stratum."""
        for t in ("Marcus Thorne is a journalist.",
                  "This fact was sourced by Marcus Thorne.",
                  "Marcus Thorne said the letter would be ready by night.",
                  "I am Marcus Thorne."):
            self.assertTrue(self.leaked(t), t)

    def test_name_leaked_excludes_other_people_and_absence(self):
        for t in ("This fact was sourced by Marcus Eldridge.",
                  "Alice, Bob and Carol went to the shop.",
                  "The capital of France is Paris."):
            self.assertFalse(self.leaked(t), t)

    def test_every_assertion_also_leaks(self):
        """A claim to be the subject necessarily mentions the subject; the
        reverse does not hold. Pinned so the two can never invert."""
        for t in ("I'm Marcus.", "My full name is Marcus Thorne.",
                  "Marcus Thorne.", "As Marcus Thorne, I can help."):
            self.assertTrue(self.leaked(t), t)
            self.assertTrue(self.v2(t), t)
