prompt_template = """
You are an expert at creating questions based on coding materials and documentation.
Your goal is to prepare a coder or programmer for their exam and coding tests.
You do this by asking questions about the text below:

----------------
{text}
----------------

Create questions that will prepare the coders or programmers for their tests.
Make sure not to lose any important information.

QUESTIONS:
"""

refine_template = """
You are an expert at creating practise questions based on coding materials and documentation.
Your goal is to help coder or programmaer prepare for a coding test.
We have received some practise questions to a certain extent: {existing_answer}
We have the options to refine the existing questions or add new ones.
(only if necessary) with some more context below.

-------------
{text}
-------------

Given the new context, refine the original questions in English.
If the context is not helpful, please provide the original questions.
QUESTIONS:
"""