Small Sentence Tokenizer
===

A sentence tokenizer model takes in two things, 'left', and 'right'.  Left is the sentence up to the final character.  Right is everything after the final character.  If sentence tokenizer returns 'true', then the final character of the left sentence marks a complete sentence.  We are avoiding 'head', 'tail', 'start', and 'end' because these are ambiguous when it comes to the boundaries of sentences.  (Is 'end' the end of the first sentence or does it refer to the second part of ths string?)

Examples:
| -Left- | -Right- | -SentTokenizerOut:- |
| I was born in Washington D.C. | , but I grew up in Los Angeles. | False |
| I was born in Washington D.C. |  I grew up in Los Angeles. | True |
| This sentence is sp | lit in the middle of a word, so obviously it's not a sentence. | False |
| Have you ever had a dream? | That you were so sure was real? | True |
| Have you ever had a dream | that you were so sur was real? | False |
| "Welcome, Doctor!" | he said. | False |
| "Welcome, Doctor!" | (empty) | True |
| "Welcome, " | (empty) | False |

