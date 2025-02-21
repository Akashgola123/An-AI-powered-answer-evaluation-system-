import setuptools

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

__version__ = "0.0.1"

REPO_NAME = "An-AI-powered-answer-evaluation-system-"
AUTHOR_USER_NAME = "Akashgola123"
SRC_REPO = "AIAnswerEvaluationSystem"  # Removed spaces
AUTHOR_EMAIL = "golaakash0@gmail.com"

setuptools.setup(
    name=SRC_REPO,
    version=__version__,
    author=AUTHOR_USER_NAME,
    author_email=AUTHOR_EMAIL,
    description="An AI-powered answer evaluation system using RAG and LLMs",
    long_description=long_description,
    long_description_content_type="text/markdown",  # Fixed content type parameter
    url=f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",
    project_urls={
        "Bug Tracker": f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}/issues",
    },
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src")
)
