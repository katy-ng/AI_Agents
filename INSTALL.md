# Setting up, from nothing to a working notebook

Written for students. Windows instructions; the macOS and Linux differences are noted at each step.
Allow about half an hour, most of it downloads. Work through it in order and run the check at the end.

## What you are installing

Four things: Python and a set of packages, the literature store itself, an API key so a language model
will answer, and one embedding model that turns text into numbers. Nothing here writes to the store, and
nothing you do can damage it.

## 1. Python

You need Python 3.11 or newer. Check what you have:

```powershell
python --version
```

If that fails or reports something older, install Python 3.13 from python.org and tick "Add Python to
PATH" during installation.

## 2. The code

Clone the repository, or unzip the copy you were given. The examples below assume it sits at
`D:\rag\rag_development`; substitute your own path throughout.

```powershell
git clone https://github.com/davidmckinnon/rag-development.git D:\rag\rag_development
```

The repository is private, so ask David to add your GitHub account first.

## 3. A virtual environment

A virtual environment is a private set of packages, so this project cannot break anything else on your
machine. Create one next to the code, not inside it:

```powershell
python -m venv D:\rag\rag_env
```

From here on, always call that environment's Python by its full path rather than typing `python`. It is
more typing and it removes an entire class of confusion about which Python is running.

On macOS or Linux the interpreter is at `rag_env/bin/python` rather than `rag_env\Scripts\python.exe`.

## 4. The packages

Install PyTorch first, on its own, because the default package would try to fetch a version built for
NVIDIA graphics cards. Unless you have one, you want the processor-only build, which is much smaller:

```powershell
D:\rag\rag_env\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

If you do have an NVIDIA card and want to use it, drop the `--index-url` line and install `torch` plainly.
Nothing in the notebook needs a graphics card. It embeds one short question per query.

Then the rest:

```powershell
D:\rag\rag_env\Scripts\python.exe -m pip install sentence-transformers numpy pandas requests lxml pymupdf ipykernel ipywidgets jupyter
```

`ipywidgets` is there to stop a warning about progress bars on the first cell. `lxml` and `pymupdf` are
used by the paper-fetching code, which the notebook imports but does not run.

## 5. Register the environment with Jupyter

So that the notebook can find this environment as a kernel:

```powershell
D:\rag\rag_env\Scripts\python.exe -m ipykernel install --user --name rag_env --display-name "Python (rag_env)"
```

When you open the notebook, choose **Python (rag_env)** as the kernel. Getting this wrong is the single
most common setup failure: the notebook runs but cannot find the project's code.

## 6. The store

You will be given `students-v1.zip`, about 190 MB. Unzip it somewhere outside the repository, for example
to `D:\rag\stores\students-v1`. It must end up containing `store.sqlite` and `vectors.npy`.

It holds 3,874 papers as about 60,000 searchable pieces. Papers published under an open licence are held
in full; everything else is held as its abstract alone, which is why many citations you see will carry the
caveat "abstract only". That is correct behaviour, not a fault.

## 7. An API key

The notebook sends your question and the retrieved text to a language model through OpenRouter, which is a
single gateway to many models. You need your own key, and you pay for your own usage. A question costs
roughly two to four cents.

1. Create an account at openrouter.ai.
2. Add a small amount of credit. Five dollars is plenty for this course.
3. Under Keys, create a key and copy it. It begins `sk-or-`.
4. Set a spending limit on the key while you are creating it. This is worth doing: a loop that runs away
   costs money, and a limit turns that into an error message instead of a bill.

Now set it as an environment variable, so it never appears in any file:

```powershell
[Environment]::SetEnvironmentVariable("OPENROUTER_API_KEY", "sk-or-your-key-here", "User")
```

On macOS or Linux, add `export OPENROUTER_API_KEY="sk-or-..."` to your shell profile.

**Close and reopen your terminal and your editor afterwards.** Environment variables are read when a
program starts, so a program already running will not see the new value. This catches almost everyone.

Never paste the key into a notebook cell, a file, or a message. Anything committed to git is permanent,
and a key in a repository has to be revoked rather than deleted.

## 8. Tell the code which store to read

```powershell
[Environment]::SetEnvironmentVariable("SCRAG_STORE", "D:\rag\stores\students-v1", "User")
```

Reopen your terminal and editor again. If this is not set, the code quietly looks in a folder that may not
exist or may be mid-rebuild, and you will get confusing results rather than a clear error.

## 9. Fetch the embedding model once

The store's text was turned into numbers by a specific model at a specific version, and your questions must
use the same one. Fetch it now, so that later runs need no network:

```powershell
D:\rag\rag_env\Scripts\python.exe -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-en-v1.5', revision='5c38ec7c405ec4b44b94cc5a9bb96e735b38267a')"
```

About 130 MB. It goes into a cache in your user folder, not into the project.

## 10. Check it all worked

```powershell
D:\rag\rag_env\Scripts\python.exe -c "import os, sqlite3, sentence_transformers, requests; s=os.environ['SCRAG_STORE']; m=dict(sqlite3.connect(s+'/store.sqlite').execute('select key,value from meta')); print('store  :', s); print('papers :', m['papers'], 'chunks:', m['chunks']); print('key    :', 'set' if os.environ.get('OPENROUTER_API_KEY') else 'NOT SET')"
```

You should see a store path, a paper and chunk count, and `key: set`. If any line is wrong, fix it before
opening the notebook; every later error will be a consequence of it.

Then open `notebooks/tutorial_1.ipynb`, choose the **Python (rag_env)** kernel, and run the first cell.

## When something goes wrong

**`No module named 'scrag'`** — the kernel is not the one you registered, or the notebook is not putting the
repository root on the path. Check the kernel name in the top right first.

**`KeyError: 'OPENROUTER_API_KEY'`** — the variable is set but your editor was already running when you set
it. Close it completely and reopen.

**`unable to open database file`** — `SCRAG_STORE` points somewhere without a `store.sqlite`. Check the path,
and check the zip unpacked into the folder you think it did rather than into a folder inside it.

**A 402 or credit error from OpenRouter** — the key has no credit, or you hit your own spending limit.

**It is very slow the first time** — the embedding model loads on the first question, about ten seconds.
Later questions in the same session are quick.
