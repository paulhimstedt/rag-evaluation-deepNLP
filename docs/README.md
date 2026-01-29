# RAG Evaluation Documentation

Complete documentation for the RAG evaluation workflow.

---

## 🚀 Getting Started

**New to this project?** Start here:

1. **[QUICK_REFERENCE.md](QUICK_REFERENCE.md)** - 1-page quick start guide
2. **[DATASET_PREPARATION_GUIDE.md](DATASET_PREPARATION_GUIDE.md)** - Complete step-by-step instructions
3. **[WORKFLOW_DIAGRAM.md](WORKFLOW_DIAGRAM.md)** - Visual workflow diagrams

---

## 📚 Documentation Index

### Core Guides

| Document | Purpose | Audience |
|----------|---------|----------|
| **[QUICK_REFERENCE.md](QUICK_REFERENCE.md)** | 1-page quick start with common commands | Everyone |
| **[DATASET_PREPARATION_GUIDE.md](DATASET_PREPARATION_GUIDE.md)** | Complete workflow with troubleshooting | All users |
| **[WORKFLOW_DIAGRAM.md](WORKFLOW_DIAGRAM.md)** | Visual diagrams of the process | Visual learners |

### Setup & Configuration

| Document | Purpose | Audience |
|----------|---------|----------|
| **[QUICKSTART.md](QUICKSTART.md)** | Original quick start guide | First-time users |
| **[ENV_SETUP.md](ENV_SETUP.md)** | Environment setup details | System administrators |
| **[MSMARCO_SETUP.md](MSMARCO_SETUP.md)** | MS-MARCO specific instructions | Users needing MS-MARCO |

### Reference & Status

| Document | Purpose | Audience |
|----------|---------|----------|
| **[DATASET_STATUS.md](DATASET_STATUS.md)** | Current status of all 7 datasets | All users |
| **[DATASET_ISSUES.md](DATASET_ISSUES.md)** | Known dataset issues and workarounds | Troubleshooting |
| **[TROUBLESHOOTING.md](TROUBLESHOOTING.md)** | Common problems and solutions | Problem solvers |
| **[README_TRANSFORMERS_ORIGINAL.md](README_TRANSFORMERS_ORIGINAL.md)** | Legacy README from the original RAG research project | Reference |

### Technical Details

| Document | Purpose | Audience |
|----------|---------|----------|
| **[IMPLEMENTATION_COMPLETE.md](IMPLEMENTATION_COMPLETE.md)** | Complete technical implementation | Developers |
| **[IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)** | Implementation overview | Technical leads |
| **[DATASET_FIXES_SUMMARY.md](DATASET_FIXES_SUMMARY.md)** | History of dataset fixes | Maintainers |
| **[MODAL_EVALUATION.md](MODAL_EVALUATION.md)** | Modal-specific evaluation details | Modal users |

---

## 🎯 Quick Navigation

### I want to...

**...get started quickly**
→ [QUICK_REFERENCE.md](QUICK_REFERENCE.md)

**...understand the complete workflow**
→ [DATASET_PREPARATION_GUIDE.md](DATASET_PREPARATION_GUIDE.md)

**...see visual diagrams**
→ [WORKFLOW_DIAGRAM.md](WORKFLOW_DIAGRAM.md)

**...set up MS-MARCO**
→ [MSMARCO_SETUP.md](MSMARCO_SETUP.md)

**...check dataset status**
→ [DATASET_STATUS.md](DATASET_STATUS.md)

**...troubleshoot issues**
→ [TROUBLESHOOTING.md](TROUBLESHOOTING.md)

**...understand technical details**
→ [IMPLEMENTATION_COMPLETE.md](IMPLEMENTATION_COMPLETE.md)

---

## 📖 Documentation Overview

### Workflow Documents (Start Here!)

#### [QUICK_REFERENCE.md](QUICK_REFERENCE.md)
**Single-page reference for the entire workflow**
- Prerequisites checklist
- 3-command workflow
- Common commands
- Troubleshooting quick fixes
- Expected results

**Best for**: Quick lookup, sharing with colleagues

#### [DATASET_PREPARATION_GUIDE.md](DATASET_PREPARATION_GUIDE.md)
**Complete step-by-step guide**
- Why this approach works
- Prerequisites setup
- Step-by-step workflow
- Team collaboration guide
- Updating datasets
- Advanced usage

**Best for**: First-time setup, comprehensive understanding

#### [WORKFLOW_DIAGRAM.md](WORKFLOW_DIAGRAM.md)
**Visual representation of the workflow**
- Architecture diagrams
- Data flow charts
- Timeline visualizations
- Team collaboration scenarios
- File format examples

**Best for**: Visual learners, presentations, onboarding

---

### Setup & Configuration

#### [QUICKSTART.md](QUICKSTART.md)
Original quick start guide with evaluation basics.

#### [ENV_SETUP.md](ENV_SETUP.md)
Environment setup including Python, Modal, and dependencies.

#### [MSMARCO_SETUP.md](MSMARCO_SETUP.md)
**Comprehensive MS-MARCO setup guide**
- Why MS-MARCO requires special handling
- Kaggle API setup (automatic download)
- Manual download instructions
- Modal secret configuration
- Troubleshooting

**Best for**: Users who need MS-MARCO dataset

---

### Status & Issues

#### [DATASET_STATUS.md](DATASET_STATUS.md)
**Current status of all 7 datasets**
- Which datasets work (5-6/7)
- Why some don't work
- Manual workarounds
- Technical constraints
- Impact assessment

**Best for**: Understanding dataset availability

#### [DATASET_ISSUES.md](DATASET_ISSUES.md)
**Known issues and solutions**
- Dataset-by-dataset issue list
- Root causes
- Attempted solutions
- Workarounds

**Best for**: Investigating specific dataset problems

#### [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
**Solutions to common problems**
- Error messages and fixes
- Environment issues
- Upload/download problems
- Modal-specific issues

**Best for**: Solving problems quickly

---

### Technical Documentation

#### [IMPLEMENTATION_COMPLETE.md](IMPLEMENTATION_COMPLETE.md)
**Complete technical implementation details**
- What was built
- How it works
- Architecture overview
- Technical decisions
- Testing approach
- Maintenance guide

**Best for**: Developers, maintainers, technical understanding

#### [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)
**High-level implementation overview**
- Key components
- Design decisions
- Integration points

**Best for**: Technical leads, architecture review

#### [DATASET_FIXES_SUMMARY.md](DATASET_FIXES_SUMMARY.md)
**History of dataset fixes**
- What issues were found
- How they were fixed
- Lessons learned

**Best for**: Understanding evolution, avoiding past mistakes

#### [MODAL_EVALUATION.md](MODAL_EVALUATION.md)
**Modal-specific evaluation details**
- Modal configuration
- Resource requirements
- Performance tuning

**Best for**: Modal platform optimization

---

## 🔍 Finding Information

### By Task

| Task | Document |
|------|----------|
| First-time setup | [QUICK_REFERENCE.md](QUICK_REFERENCE.md) + [DATASET_PREPARATION_GUIDE.md](DATASET_PREPARATION_GUIDE.md) |
| Quick command lookup | [QUICK_REFERENCE.md](QUICK_REFERENCE.md) |
| Troubleshooting errors | [TROUBLESHOOTING.md](TROUBLESHOOTING.md) |
| Setting up MS-MARCO | [MSMARCO_SETUP.md](MSMARCO_SETUP.md) |
| Checking dataset status | [DATASET_STATUS.md](DATASET_STATUS.md) |
| Understanding architecture | [WORKFLOW_DIAGRAM.md](WORKFLOW_DIAGRAM.md) + [IMPLEMENTATION_COMPLETE.md](IMPLEMENTATION_COMPLETE.md) |
| Team onboarding | [QUICK_REFERENCE.md](QUICK_REFERENCE.md) |

### By Role

| Role | Recommended Docs |
|------|------------------|
| **New User** | QUICK_REFERENCE → DATASET_PREPARATION_GUIDE → WORKFLOW_DIAGRAM |
| **Colleague** | QUICK_REFERENCE (that's it!) |
| **Developer** | IMPLEMENTATION_COMPLETE → IMPLEMENTATION_SUMMARY → DATASET_ISSUES |
| **Troubleshooter** | TROUBLESHOOTING → DATASET_ISSUES → DATASET_STATUS |
| **Team Lead** | WORKFLOW_DIAGRAM → IMPLEMENTATION_COMPLETE → QUICK_REFERENCE |

---

## 📝 Documentation Conventions

### Links
- All internal links use relative paths (work within docs/)
- Links to parent directory files use `../filename`

### Code Blocks
- Bash commands: Show working directory context
- Python code: Include imports and error handling
- Output: Show expected results with `# Expected output` comments

### Status Indicators
- ✅ Working / Complete
- ⚠️ Partial / Needs attention
- ❌ Not working / Unavailable
- 🚧 Work in progress

---

## 🔄 Keeping Documentation Updated

When making changes to the workflow:

1. **Update core guides first**: QUICK_REFERENCE, DATASET_PREPARATION_GUIDE, WORKFLOW_DIAGRAM
2. **Update status docs**: DATASET_STATUS, TROUBLESHOOTING
3. **Update technical docs**: IMPLEMENTATION_COMPLETE
4. **Test all commands** in documentation
5. **Verify all links** work correctly

---

## 📦 Documentation Structure

```
docs/
├── README.md (this file)
│
├── Quick Start (Start Here!)
│   ├── QUICK_REFERENCE.md
│   ├── DATASET_PREPARATION_GUIDE.md
│   └── WORKFLOW_DIAGRAM.md
│
├── Setup & Config
│   ├── QUICKSTART.md
│   ├── ENV_SETUP.md
│   └── MSMARCO_SETUP.md
│
├── Status & Issues
│   ├── DATASET_STATUS.md
│   ├── DATASET_ISSUES.md
│   └── TROUBLESHOOTING.md
│
└── Technical
    ├── IMPLEMENTATION_COMPLETE.md
    ├── IMPLEMENTATION_SUMMARY.md
    ├── DATASET_FIXES_SUMMARY.md
    └── MODAL_EVALUATION.md
```

---

## 🚀 Ready to Start?

1. **[QUICK_REFERENCE.md](QUICK_REFERENCE.md)** - Get started in 3 commands
2. **[DATASET_PREPARATION_GUIDE.md](DATASET_PREPARATION_GUIDE.md)** - Complete instructions
3. **[WORKFLOW_DIAGRAM.md](WORKFLOW_DIAGRAM.md)** - Understand the flow

Have questions? Check **[TROUBLESHOOTING.md](TROUBLESHOOTING.md)**
