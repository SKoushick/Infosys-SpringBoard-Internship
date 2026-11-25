"""
Generate the 5-minute professional speaker script for TEDMind
in both DOCX format and Markdown format.
"""

import os
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DOCS_DIR = os.path.join(REPO_ROOT, "docs")
OUTPUT_DIR = os.path.join(DOCS_DIR, "presentation")

DOCX_PATH = os.path.join(OUTPUT_DIR, "TEDMind_5_Minute_Speaker_Script.docx")
MD_PATH = os.path.join(OUTPUT_DIR, "TEDMind_5_Minute_Speaker_Script.md")

SCRIPT_DATA = [
    {
        "slide_num": 1,
        "title": "Title Slide — Welcome & Project Overview",
        "time": "0:00 – 0:30 (30 seconds)",
        "script": (
            "Good morning, respected mentors and evaluation panel. My name is S Koushick, "
            "and today I am privileged to present TEDMind: an AI-powered personalized TED Talks recommendation "
            "and goal-oriented learning platform, developed as the capstone engineering deliverable for "
            "the Infosys Springboard Internship 7.0.\n\n"
            "Our guiding philosophy is simple: 'Discover. Focus. Learn. Grow.' "
            "TEDMind bridges the gap between passive video streaming and intentional educational pathways by uniting "
            "classical lexical retrieval, dense neural semantic transformers, and algorithmic diversity re-ranking "
            "across 2,550 presentations from the global TED catalog."
        ),
        "transition": "Let us examine the fundamental real-world challenge that motivated this system."
    },
    {
        "slide_num": 2,
        "title": "Problem Statement — Friction in Self-Directed Learning",
        "time": "0:30 – 1:00 (30 seconds)",
        "script": (
            "Online learners face severe discovery friction. In a sprawling catalog like TED, viewers experience "
            "severe choice paralysis—navigating over 2,500 interdisciplinary talks without a structured path.\n\n"
            "Furthermore, traditional search suffers from vocabulary mismatch: keyword search retrieves only literal "
            "word matches, completely missing relevant presentations that express the same concept using alternative vocabulary. "
            "Finally, standard content-based algorithms trap users in narrow filter bubbles by repeatedly recommending "
            "near-identical topics, with zero evaluation of whether a talk is foundational or advanced."
        ),
        "transition": "This motivated us to build TEDMind as a multi-signal, goal-oriented solution."
    },
    {
        "slide_num": 3,
        "title": "Project Objectives & Solution Overview",
        "time": "1:00 – 1:30 (30 seconds)",
        "script": (
            "TEDMind is designed for students, researchers, and professional lifelong learners. "
            "Instead of treating recommendations as isolated entertainment suggestions, TEDMind provides three core innovations:\n\n"
            "First, multi-signal blending: we combine exact keyword matching with deep semantic context. "
            "Second, diversity optimization via Maximal Marginal Relevance, ensuring that recommended slates are "
            "broad and serendipitous rather than repetitive. "
            "And third, transparent Explainable AI badges, providing clear reasons for every suggestion so learners understand "
            "why a talk was recommended."
        ),
        "transition": "To achieve this, we began with rigorous data collection and preprocessing in Modules 1 and 2."
    },
    {
        "slide_num": 4,
        "title": "Dataset & Preprocessing (Modules 1 & 2)",
        "time": "1:30 – 2:00 (30 seconds)",
        "script": (
            "Our canonical dataset encompasses 2,550 curated presentations published between 2006 and 2020, "
            "combining rich metadata with full speech transcripts. In Module 1, we validated complete catalog integrity, "
            "resolving missing IDs and pruning broken cross-references.\n\n"
            "In Module 2, we conducted text sanitization: we engineered regular expressions to strip stage cues such as "
            "laughter and applause from transcripts, normalized Unicode entities, and for talks lacking transcripts, "
            "we synthesized a high-density 'content soup' weighting titles, speakers, tags, and descriptions. "
            "This ensured zero catalog talks had to be discarded."
        ),
        "transition": "With a pristine text corpus, we architected our dual-space vector representation in Module 3."
    },
    {
        "slide_num": 5,
        "title": "Feature Engineering & Vectorization (Module 3)",
        "time": "2:00 – 2:30 (30 seconds)",
        "script": (
            "Recommenders often fail when relying on a single vector space. In Module 3, we engineered a dual-space architecture.\n\n"
            "In the sparse lexical space, a sublinear TF-IDF vectorizer extracts 44,613 unigram and bigram features. "
            "This guarantees laser precision on specialized technical terms, acronyms, and proper speaker names.\n\n"
            "In parallel, in the dense semantic space, we utilized Sentence-BERT using the all-MiniLM-L6-v2 transformer "
            "to project talks into 384 continuous dimensions. This captures nuanced philosophical concepts, thematic synonyms, "
            "and conceptual paraphrasing that keyword search misses completely."
        ),
        "transition": "These two spaces feed directly into our multi-signal recommendation engine in Module 4."
    },
    {
        "slide_num": 6,
        "title": "Recommendation Engine & Diversification (Module 4)",
        "time": "2:30 – 3:15 (45 seconds)",
        "script": (
            "Our recommendation engine blends four distinct signals into a unified hybrid score: 35% lexical TF-IDF, "
            "45% dense Sentence-BERT similarity, 15% curated taxonomy tag overlap, and 5% speaker affinity, "
            "with dynamic exclusion of user-disliked talks.\n\n"
            "To solve the filter-bubble dilemma, we applied Maximal Marginal Relevance—or MMR—with a tuned lambda of 0.70. "
            "MMR penalizes redundant candidate talks that are overly similar to presentations already selected for the slate. "
            "This delivers an outstanding 20.6% boost in intra-list diversity while preserving ranking precision, "
            "all computed in sub-60 millisecond latency."
        ),
        "transition": "Let us examine our empirical benchmark results evaluated in Module 5."
    },
    {
        "slide_num": 7,
        "title": "Empirical Evaluation & Results (Module 5)",
        "time": "3:15 – 3:45 (30 seconds)",
        "script": (
            "In Module 5, we benchmarked our models across 100 standardized evaluation queries under identical conditions. "
            "As shown in our scorecard, the Hybrid Multi-Signal model achieved the highest overall ranking accuracy, "
            "with an NDCG@10 of 0.0076 and MAP@10 of 0.0035, outperforming standalone lexical and semantic baselines.\n\n"
            "Crucially, introducing MMR expanded Intra-List Diversity to 0.6953—a 20.6% increase—while expanding catalog coverage "
            "to 54.16%. In accordance with scientific integrity, we note that ground truth is computed against TED's curated "
            "companion talk proxy, and qualitative human evaluation is formally cataloged as pending."
        ),
        "transition": "Now, let us view how this intelligence is delivered in our production web application in Module 6."
    },
    {
        "slide_num": 8,
        "title": "Web Application Architecture & UI (Module 6)",
        "time": "3:45 – 4:15 (30 seconds)",
        "script": (
            "Module 6 delivers a full-stack Django 5.2 application styled with modern Tailwind CSS. "
            "The slide displays genuine live screenshots captured from our running server.\n\n"
            "Learners can explore all 2,550 presentations in our catalog explorer, dive into talk details with synchronized "
            "speech transcripts, and interact with our Recommendation Studio by providing seed talks, curiosity prompts, or topic themes. "
            "Guest sessions seamlessly migrate bookmarks upon user login, and authentication gateways securely protect "
            "private collections and administrative controls."
        ),
        "transition": "Before concluding, let us review our engineering disclosures and Phase 2 progress."
    },
    {
        "slide_num": 9,
        "title": "Limitations, Scientific Integrity & Phase 2",
        "time": "4:15 – 4:45 (30 seconds)",
        "script": (
            "We maintain strict engineering transparency: collaborative filtering is accurately unsupported because public TED "
            "datasets lack individual user interaction IDs. Furthermore, similarity scores represent geometric vector distances "
            "rather than calibrated probabilities.\n\n"
            "In Phase 2, we progressed into pedagogical tracking. We implemented an automated complexity estimator profiling "
            "readability, lexical richness, and speaking pace into foundational, intermediate, and specialized stages. "
            "We also built transactional path deletion services that guarantee learner watch-time and ratings are never orphaned or lost."
        ),
        "transition": "To summarize our journey and practical contributions in Module 7."
    },
    {
        "slide_num": 10,
        "title": "Conclusion & Practical Impact (Module 7)",
        "time": "4:45 – 5:15 (30 seconds)",
        "script": (
            "In conclusion, TEDMind represents an end-to-end engineering achievement across all seven Infosys Springboard modules. "
            "With 110 automated tests passing at a 100% rate, robust database referential integrity, and complete technical documentation, "
            "we have demonstrated how recommender systems can evolve from passive entertainment feeds into intentional, "
            "diversity-aware educational platforms.\n\n"
            "I extend my sincere gratitude to our Infosys Springboard mentors. Thank you, and I look forward to your questions."
        ),
        "transition": "End of Presentation — Q&A Session."
    }
]

def build_docx():
    doc = Document()
    
    # Page Setup
    sections = doc.sections
    for section in sections:
        section.top_margin = Inches(1.0)
        section.bottom_margin = Inches(1.0)
        section.left_margin = Inches(1.0)
        section.right_margin = Inches(1.0)
        
    # Title
    title = doc.add_paragraph()
    r_title = title.add_run("TEDMind: 5-Minute Master Speaker Script")
    r_title.bold = True
    r_title.font.name = "Segoe UI"
    r_title.font.size = Pt(22)
    r_title.font.color.rgb = RGBColor(230, 43, 30)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title.paragraph_format.space_after = Pt(4)
    
    # Subtitle
    sub = doc.add_paragraph()
    r_sub = sub.add_run("Infosys Springboard Internship 7.0 — Final Project Review Presentation Script\nAuthor: S Koushick  |  Pacing: ~135 WPM  |  Total Speaking Time: 5:15 minutes")
    r_sub.font.name = "Segoe UI"
    r_sub.font.size = Pt(11)
    r_sub.font.color.rgb = RGBColor(71, 85, 105)
    sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    sub.paragraph_format.space_after = Pt(20)
    
    # Speaker Instructions Box
    tip = doc.add_paragraph()
    r_tip = tip.add_run("Presentation Delivery Guidelines:\n• Speak in a natural, conversational, confident tone.\n• Total word count is approximately 680 words, comfortably fitting a 5-minute speaking slot.\n• Use the slide transition cues to seamlessly advance your slides.\n• Keep eye contact with the evaluation panel.")
    r_tip.italic = True
    r_tip.font.name = "Segoe UI"
    r_tip.font.size = Pt(10)
    r_tip.font.color.rgb = RGBColor(30, 41, 59)
    tip.paragraph_format.space_after = Pt(24)
    
    for item in SCRIPT_DATA:
        # Heading
        h = doc.add_paragraph()
        r_h = h.add_run(f"Slide {item['slide_num']}: {item['title']}")
        r_h.bold = True
        r_h.font.name = "Segoe UI"
        r_h.font.size = Pt(14)
        r_h.font.color.rgb = RGBColor(15, 23, 42)
        h.paragraph_format.space_after = Pt(2)
        
        # Timing Badge
        t = doc.add_paragraph()
        r_t = t.add_run(f"Allocated Timing: {item['time']}")
        r_t.bold = True
        r_t.font.name = "Segoe UI"
        r_t.font.size = Pt(10)
        r_t.font.color.rgb = RGBColor(37, 99, 235)
        t.paragraph_format.space_after = Pt(6)
        
        # Script Body
        p = doc.add_paragraph()
        r_p = p.add_run(item['script'])
        r_p.font.name = "Segoe UI"
        r_p.font.size = Pt(11)
        r_p.font.color.rgb = RGBColor(51, 65, 85)
        p.paragraph_format.space_after = Pt(6)
        p.paragraph_format.line_spacing = 1.15
        
        # Transition Cue
        tr = doc.add_paragraph()
        r_tr_lbl = tr.add_run("[Slide Transition Cue]: ")
        r_tr_lbl.bold = True
        r_tr_lbl.font.name = "Segoe UI"
        r_tr_lbl.font.size = Pt(10)
        r_tr_lbl.font.color.rgb = RGBColor(230, 43, 30)
        
        r_tr = tr.add_run(item['transition'])
        r_tr.italic = True
        r_tr.font.name = "Segoe UI"
        r_tr.font.size = Pt(10)
        r_tr.font.color.rgb = RGBColor(71, 85, 105)
        tr.paragraph_format.space_after = Pt(18)
        
    doc.save(DOCX_PATH)
    print(f"Speaker script DOCX saved to: {DOCX_PATH}")

def build_markdown():
    lines = [
        "# TEDMind: 5-Minute Master Speaker Script",
        "### Infosys Springboard Internship 7.0 — Final Project Review Presentation",
        "",
        "**Author:** S Koushick  ",
        "**Platform:** TEDMind (*Discover. Focus. Learn. Grow.*)  ",
        "**Total Pacing:** ~135 Words Per Minute  ",
        "**Total Duration:** 5 Minutes 15 Seconds  ",
        "",
        "---",
        "",
        "> [!TIP]",
        "> **Speaker Delivery Advice:** Speak at a steady, natural cadence. Emphasize transition sentences as verbal bridges before advancing each slide.",
        "",
        "---",
        ""
    ]
    
    for item in SCRIPT_DATA:
        lines.append(f"## Slide {item['slide_num']}: {item['title']}")
        lines.append(f"**Target Timing:** `{item['time']}`  ")
        lines.append("")
        lines.append(item['script'])
        lines.append("")
        lines.append(f"> **[Slide Transition Cue]:** *{item['transition']}*")
        lines.append("")
        lines.append("---")
        lines.append("")
        
    with open(MD_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Speaker script Markdown saved to: {MD_PATH}")

if __name__ == "__main__":
    build_docx()
    build_markdown()
