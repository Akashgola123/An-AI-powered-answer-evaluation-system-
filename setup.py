import setuptools

with open("README.md", "r",encoding="utf-8") as fh:
    long_description = fh.read()


__version__ = "0.0.0"

REPO_NAME = "An-AI-powered-answer-evaluation-system-"
AUTHOR_USER_NAME = "Akashgola123"
SRC_REPO = "AI_Assisted_Assessment_System"
AUTHOR_EMAIL = "golaakash0@gmail.com"


setuptools.setup(
    name=f"{REPO_NAME}-{SRC_REPO}",
    version=__version__,
    author=f"{AUTHOR_USER_NAME}",
    author_email=f"{AUTHOR_EMAIL}",
    description="An AI-powered answer evaluation system using RAG and LLMs to assess student responses by retrieving knowledge from PDFs and question banks.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url = f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",
    project_urls={
        "Bug Tracker": f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}/issues",
    },
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src")

)