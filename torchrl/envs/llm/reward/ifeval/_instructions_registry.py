# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


# Original LICENSE:
#  Origin: https://github.com/google-research/google-research/blob/master/instruction_following_eval/instruction_following_eval.py
# coding=utf-8
# Copyright 2024 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from __future__ import annotations

from ._instructions import (
    BulletListChecker,
    CapitalLettersEnglishChecker,
    CapitalWordFrequencyChecker,
    CommaChecker,
    ConstrainedResponseChecker,
    EndChecker,
    ForbiddenWords,
    HighlightSectionChecker,
    JsonFormat,
    KeywordChecker,
    KeywordFrequencyChecker,
    LetterFrequencyChecker,
    LowercaseLettersEnglishChecker,
    NumberOfSentences,
    NumberOfWords,
    ParagraphChecker,
    ParagraphFirstWordCheck,
    PlaceholderChecker,
    PostscriptChecker,
    QuotationChecker,
    RepeatPromptThenAnswer,
    ResponseLanguageChecker,
    SectionChecker,
    TitleChecker,
    TwoResponsesChecker,
)

_KEYWORD = "keywords:"
_LANGUAGE = "language:"
_LENGTH = "length_constraints:"
_CONTENT = "detectable_content:"
_FORMAT = "detectable_format:"
_MULTITURN = "multi-turn:"
_COMBINATION = "combination:"
_STARTEND = "startend:"
_CHANGE_CASES = "change_case:"
_PUNCTUATION = "punctuation:"

INSTRUCTION_DICT = {
    _KEYWORD + "existence": KeywordChecker,
    _KEYWORD + "frequency": KeywordFrequencyChecker,
    # TODO(jeffreyzhou): make a proper set of sentences to choose from
    # _KEYWORD + "key_sentences": KeySentenceChecker,
    _KEYWORD + "forbidden_words": ForbiddenWords,
    _KEYWORD + "letter_frequency": LetterFrequencyChecker,
    _LANGUAGE + "response_language": ResponseLanguageChecker,
    _LENGTH + "number_sentences": NumberOfSentences,
    _LENGTH + "number_paragraphs": ParagraphChecker,
    _LENGTH + "number_words": NumberOfWords,
    _LENGTH + "nth_paragraph_first_word": ParagraphFirstWordCheck,
    _CONTENT + "number_placeholders": PlaceholderChecker,
    _CONTENT + "postscript": PostscriptChecker,
    _FORMAT + "number_bullet_lists": BulletListChecker,
    # TODO(jeffreyzhou): Pre-create paragraph or use prompt to replace
    # _CONTENT + "rephrase_paragraph": RephraseParagraph,
    _FORMAT + "constrained_response": ConstrainedResponseChecker,
    _FORMAT + "number_highlighted_sections": HighlightSectionChecker,
    _FORMAT + "multiple_sections": SectionChecker,
    # TODO(tianjianlu): Re-enable rephrasing with preprocessing the message.
    # _FORMAT + "rephrase": RephraseChecker,
    _FORMAT + "json_format": JsonFormat,
    _FORMAT + "title": TitleChecker,
    # TODO(tianjianlu): Re-enable with specific prompts.
    # _MULTITURN + "constrained_start": ConstrainedStartChecker,
    _COMBINATION + "two_responses": TwoResponsesChecker,
    _COMBINATION + "repeat_prompt": RepeatPromptThenAnswer,
    _STARTEND + "end_checker": EndChecker,
    _CHANGE_CASES + "capital_word_frequency": CapitalWordFrequencyChecker,
    _CHANGE_CASES + "english_capital": CapitalLettersEnglishChecker,
    _CHANGE_CASES + "english_lowercase": LowercaseLettersEnglishChecker,
    _PUNCTUATION + "no_comma": CommaChecker,
    _STARTEND + "quotation": QuotationChecker,
}
