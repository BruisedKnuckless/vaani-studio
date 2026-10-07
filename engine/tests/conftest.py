import os
import tempfile

# Keep tests away from the real Application Support and Music folders.
_tmp = tempfile.mkdtemp(prefix="vaani-tests-")
os.environ["VAANI_HOME"] = os.path.join(_tmp, "home")
os.environ["VAANI_OUTPUT"] = os.path.join(_tmp, "out")
for _env in ("VAANI_SARVAM_API_KEY", "VAANI_OPENAI_API_KEY", "VAANI_ANTHROPIC_API_KEY", "VAANI_HF_TOKEN"):
    os.environ.pop(_env, None)
