1. Summary: Layer vs. Head
It is easy to confuse these two, so here is the distinction:

Heads (Horizontal): Happen simultaneously. 12 heads look at the sentence from 12 different "angles" at the exact same time. (12 attentions +The Feed-Forward Network)

Layers (Vertical): Happen sequentially. Layer 2 cannot start until Layer 1 is finished. 

2. BERT --Bidirectional Encoder Representations from Transformers
The BERT Way: Bidirectional (All at once)In a bidirectional model like BERT, the model uses the Self-Attention mechanism we discussed earlier ($Q, K, V$) to look at every word in the sentence at the same time.For the word "bank":It looks Left: It sees [The].It looks Right: It sees [of, the, river, was, muddy].

3. overrid changes 

<!-- import importlib
import src.llm_client
importlib.reload(src.llm_client)
from src.llm_client import LLMClient  # re-import! -->

OR
%load_ext autoreload
%autoreload 

4. show off new created kernel
    ctrl+shift +P --> developer refresh reload, mannually input address, register 
5. google ai ---recommanded model :gemini-2.5-flash
# 1. Remove the old "Developer" SDK to clear the path
pip uninstall google-generativeai -y

# 2. Install the NEW Unified SDK
pip install google-genai

# 3. Refresh your environment info
pip install --upgrade google-genai

5. Cached encoder (avoid reloading on every call)
 if encoding_name not in _tiktoken_cache:
        try:
            import tiktoken
            _tiktoken_cache[encoding_name] = tiktoken.get_encoding(encoding_name)
        except ImportError:
            raise ImportError("Install tiktoken: pip install tiktoken")

    return _tiktoken_cache[encoding_name]

6. how to remove duplicate from list and in order--dict.fromkeys()
abs_links = list(dict.fromkeys(parser.abs_links))[:max_papers]

7. {text[i:i+k] for i in range(len(text) - k + 1)} type :set
 for not contain duplicate value 
 set(str)

 8. git checkout -b Homework-Week2 
 Create a new branch called Homework-Week2
Switch to it immediately
9.git branch -m Homework-Week2
rename 

10. How to submit homework to git
0. One-time setup (already done)
✅ Fork teacher repo on GitHub
✅ Clone your fork
✅ Add upstream  git remote add upstream https://github.com/teacher/repo.git
    git remote -v
        origin   → your repo
        upstream → teacher repo
 WEEKLY WORKFLOW (use this every assignment)       
✅ Start clean from main
git checkout main
git fetch upstream
git merge upstream/main
git push origin main



✅ Do your work
✅ Create a new branch
git checkout -b Homework-WeekX

✅ Save your work
git add .
git commit -m "Homework Week X"

✅ Push to GitHub
git push -u origin Homework-WeekX
✅ go to branch Homework-WeekX, click on Open Pull Request and then copy  link 

### padding token vs eos token
Used to make all sequences in a batch the same length
Marks the logical end of text generation or input sequence.

### TRL
stands for Transformation Reinforcement Learning

### PEFT
stands for s