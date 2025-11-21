# Report Documentation

This guide explains how to create professional academic PDF reports using Markdown with proper citations, cross-references, figures, tables, and equations. The system uses Pandoc with pandoc-crossref and citeproc to generate beautifully formatted PDFs.

## Table of Contents

- [Building the Report](#building-the-report)
- [Document Features](#document-features)
- [Citations and Bibliography](#citations-and-bibliography)
- [Cross-References](#cross-references)
- [Figures and Images](#figures-and-images)
- [Tables](#tables)
- [Equations](#equations)
- [Sections and Structure](#sections-and-structure)
- [Text Formatting](#text-formatting)
- [Advanced Features](#advanced-features)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)

---

## Building the Report

### Quick Start

**On Linux/macOS or Windows (Git Bash):**
```bash
cd doc
./build.sh
```

**On Windows (Command Prompt):**
```bat
cd doc
build.bat
```

**Output:** `_build/report.pdf`

### What Happens During Build

The build script:
1. Checks for Docker image `aicomp-pandoc-report` (builds if needed)
2. Creates the `_build` directory
3. Runs Pandoc with:
   - pandoc-crossref filter (for cross-references)
   - citeproc filter (for citations)
   - XeLaTeX engine (for PDF generation)
4. Generates `_build/report.pdf` from `src/report.md`

### Rebuild After Changes

If the PDF is locked (open in a viewer):
```bash
cd doc
rm -f _build/report.pdf
./build.sh
```

---

## Document Features

Your generated PDF includes:

1. **Table of Contents** - Automatically generated from section headings
2. **List of Figures** - All figures with captions
3. **List of Tables** - All tables with captions
4. **Main Content** - Your markdown content
5. **Bibliography** - Automatically formatted citations in APA style

### Document Structure Example

```
Report.pdf
├── Table of Contents
├── 1. Introduction
│   └── 1.1 Background
├── 2. Methods
│   ├── 2.1 Data Collection
│   └── 2.2 Analysis
├── 3. Results
├── References
├── List of Figures
└── List of Tables
```

---

## Markdown Syntax Guide


## Citations and References

### Setting Up Your Bibliography File

Add references to `references.bib` in BibTeX format:

```bibtex
@article{smith2023,
  author = {Smith, John and Doe, Jane},
  title = {A Great Research Paper},
  journal = {Journal of Machine Learning},
  year = {2023},
  volume = {42},
  pages = {123--145}
}

@book{johnson2022,
  author = {Johnson, Alice},
  title = {Introduction to Data Science},
  publisher = {Academic Press},
  year = {2022}
}

@inproceedings{brown2021,
  author = {Brown, Bob},
  title = {Deep Learning for Sports Analytics},
  booktitle = {Proceedings of the International Conference on AI},
  year = {2021},
  pages = {45--52}
}
```

### Basic Citations

#### Parenthetical Citations
```markdown
Recent research shows promising results [@smith2023].
→ Output: Recent research shows promising results (Smith & Doe, 2023).

Multiple sources can be cited together [@smith2023; @johnson2022; @brown2021].
→ Output: Multiple sources can be cited together (Smith & Doe, 2023; Johnson, 2022; Brown, 2021).
```

#### Narrative Citations
```markdown
@smith2023 demonstrated that machine learning can...
→ Output: Smith and Doe (2023) demonstrated that machine learning can...

According to @johnson2022, data science requires...
→ Output: According to Johnson (2022), data science requires...
```

### Citations with Page Numbers and Locators

Add specific page references to your citations:

```markdown
# Single page
[@smith2023, p. 42]
→ Output: (Smith & Doe, 2023, p. 42)

# Page range
[@smith2023, pp. 42-45]
→ Output: (Smith & Doe, 2023, pp. 42-45)

# Chapter
[@johnson2022, ch. 3]
→ Output: (Johnson, 2022, ch. 3)

# Section
[@brown2021, sec. 2.1]
→ Output: (Brown, 2021, sec. 2.1)

# Figure or table
[@smith2023, fig. 5]
→ Output: (Smith & Doe, 2023, fig. 5)

# Multiple sources with page numbers
[@smith2023, pp. 15-20; @johnson2022, p. 42]
→ Output: (Smith & Doe, 2023, pp. 15-20; Johnson, 2022, p. 42)
```

#### Narrative Citations with Page Numbers
```markdown
@smith2023 [p. 42] argues that...
→ Output: Smith and Doe (2023, p. 42) argue that...
```

### References Section

The bibliography is automatically generated from cited references. Add this at the end of your markdown:

```markdown
# References {-}
```

**Note:** The `{-}` makes the section unnumbered (standard for bibliography sections).

**Features:**
- Only cited references appear in the bibliography
- References are sorted alphabetically
- Formatted in APA style (configurable via CSL file)
- Automatically includes DOIs, URLs, and access dates if provided in BibTeX

### Common Citation Patterns

```markdown
# Multiple works by same author
[@smith2023; @smith2022]
→ Output: (Smith & Doe, 2023, 2022)

# Suppress author (year only)
[-@smith2023]
→ Output: (2023)

# Multiple authors (first 3, then et al.)
Three or fewer authors: (Smith, Doe, & Brown, 2023)
More than three: (Smith et al., 2023)

# Corporate author
@{NCAA2023}
→ Output: NCAA (2023)
```

---

## Cross-References

The system uses **pandoc-crossref** to automatically generate proper cross-references with capitalization.

### Important: Automatic Labeling

**✅ Correct:** Just use `@fig:label` - the word "Figure" is added automatically
```markdown
As shown in @fig:model, we can see...
→ Output: As shown in Figure 1, we can see...
```

**❌ Wrong:** Don't write "Figure @fig:label" - it will duplicate
```markdown
As shown in Figure @fig:model...
→ Output: As shown in Figure Figure 1...  ❌
```

### Cross-Reference Types and Formatting

| Type | Markdown | Output | Parenthetical |
|------|----------|--------|---------------|
| Figure | `@fig:model` | Figure 1 | `(@fig:model)` → (Figure 1) |
| Table | `@tbl:results` | Table 1 | `(@tbl:results)` → (Table 1) |
| Equation | `@eq:loss` | Equation 1 | `(@eq:loss)` → (Equation 1) |
| Section | `@sec:intro` | Section 1 | `(@sec:intro)` → (Section 1) |

### Figure Cross-References

```markdown
# Define a figure
![Model architecture](./images/model.png){#fig:model width=70%}

# Reference it in text (various ways)
As shown in @fig:model, the architecture consists of...
→ Output: As shown in Figure 1, the architecture consists of...

See @fig:model for details.
→ Output: See Figure 1 for details.

The model architecture (@fig:model) demonstrates...
→ Output: The model architecture (Figure 1) demonstrates...

Both @fig:model and @fig:results show...
→ Output: Both Figures 1 and 2 show...
```

### Table Cross-References

```markdown
# Define a table
| Algorithm | Accuracy |
|-----------|----------|
| XGBoost   | 0.875    |

: Model performance {#tbl:results}

# Reference it
Results are shown in @tbl:results.
→ Output: Results are shown in Table 1.

See @tbl:results and @tbl:comparison.
→ Output: See Tables 1 and 2.
```

### Equation Cross-References

```markdown
# Define an equation
$$L = -\sum_{i} y_i \log(\hat{y}_i)$$ {#eq:loss}

# Reference it
Using @eq:loss, we calculate the loss.
→ Output: Using Equation 1, we calculate the loss.

Equations @eq:loss and @eq:accuracy define...
→ Output: Equations 1 and 2 define...
```

### Section Cross-References

```markdown
# Define sections with IDs
# Introduction {#sec:intro}
## Background {#sec:background}
# Methods {#sec:methods}

# Reference them
As discussed in @sec:intro, we aim to...
→ Output: As discussed in Section 1, we aim to...

See @sec:methods for implementation details.
→ Output: See Section 2 for implementation details.

Sections @sec:intro and @sec:methods describe...
→ Output: Sections 1 and 2 describe...
```

### Multiple Cross-References

```markdown
# Same type
@fig:model and @fig:results
→ Output: Figures 1 and 2

@tbl:data, @tbl:results, and @tbl:comparison
→ Output: Tables 1, 2, and 3

# Mixed types in sentence
According to @sec:methods, using @eq:loss on the data from @tbl:dataset 
produces the results in @fig:performance.
→ Output: According to Section 2, using Equation 1 on the data from 
Table 1 produces the results in Figure 1.
```

---

## Figures and Images

### Basic Figure Syntax

```markdown
![Caption text](./images/filename.png){#fig:label width=70%}
```

**Components:**
- `![]` - Markdown image syntax
- `Caption text` - Figure caption (appears in List of Figures)
- `./images/filename.png` - Path relative to `src/` directory
- `{#fig:label}` - Unique identifier for cross-referencing
- `width=70%` - Optional size specification

### Image Placement Options

#### Default Placement (Recommended)
```markdown
![Architecture overview](./images/architecture.png){#fig:arch width=70%}
```
LaTeX will place the figure optimally (here, top, bottom, or separate page).

#### Force Exact Placement
```markdown
![Important diagram](./images/diagram.png){#fig:diagram width=60% placement=H}
```
The `placement=H` forces the figure to appear exactly where you put it in the markdown.

**Placement options:**
- `H` - Exactly here (requires float package, included by default)
- `h` - Approximately here
- `t` - Top of page
- `b` - Bottom of page
- `p` - Separate page of floats
- `!` - Override LaTeX's internal parameters

**⚠️ Warning:** Using `H` too much can lead to poor page layout with large gaps. Use sparingly.

### Image Sizing

```markdown
# Percentage of text width (recommended)
![Caption](./images/img.png){#fig:id width=50%}    # Half width
![Caption](./images/img.png){#fig:id width=70%}    # 70% width (good default)
![Caption](./images/img.png){#fig:id width=100%}   # Full width

# Absolute dimensions
![Caption](./images/img.png){#fig:id width=10cm}
![Caption](./images/img.png){#fig:id height=5cm}
![Caption](./images/img.png){#fig:id width=10cm height=8cm}
```

### Multiple Images Side-by-Side

```markdown
::: {#fig:comparison}
![Model A results](./images/model_a.png){width=48%}
![Model B results](./images/model_b.png){width=48%}

Comparison of Model A and Model B performance metrics.
:::

# Reference it
The comparison in @fig:comparison shows...
→ Output: The comparison in Figure 1 shows...
```

### Subfigures with Individual Labels

```markdown
::: {#fig:models}
![Model architecture](./images/arch.png){#fig:arch width=48%}
![Training process](./images/train.png){#fig:train width=48%}

Complete model overview: (a) architecture and (b) training process.
:::

# Reference the whole figure
@fig:models shows the complete model.
→ Output: Figure 1 shows the complete model.

# Reference subfigures individually
The architecture (@fig:arch) and training process (@fig:train) are shown.
→ Output: The architecture (Figure 1a) and training process (Figure 1b) are shown.
```

### Image File Formats

Supported formats:
- **PNG** - Best for screenshots, diagrams (recommended)
- **JPEG/JPG** - Good for photographs
- **PDF** - Vector graphics (excellent for plots)
- **SVG** - Vector graphics (converted to PDF automatically)

**Tip:** Store images in `src/images/` directory for clean organization.

### Complete Figure Examples

```markdown
# Simple figure
![Loss over epochs](./images/loss_curve.png){#fig:loss width=80%}

# Figure with exact placement
![Critical diagram](./images/critical.png){#fig:critical width=70% placement=H}

# Wide figure
![Network architecture](./images/network.png){#fig:network width=100%}

# Small figure aligned
![Small icon](./images/icon.png){#fig:icon width=30%}

# Figure with height constraint
![Tall diagram](./images/tall.png){#fig:tall height=15cm}
```

---

## Tables

### Basic Table Syntax

```markdown
| Column 1      | Column 2      | Column 3      |
|---------------|---------------|---------------|
| Data 1        | Data 2        | Data 3        |
| More data 1   | More data 2   | More data 3   |

: Table caption text {#tbl:label}
```

**Components:**
- Table rows with `|` separators
- Header separator line with `-` characters
- `: Caption {#tbl:label}` - Caption and ID for cross-referencing

### Column Alignment

```markdown
| Left-aligned | Center-aligned | Right-aligned |
|:-------------|:--------------:|--------------:|
| Text         | Text           | 123.45        |
| More text    | More text      | 678.90        |
| Data         | Data           | 999.99        |

: Table with different alignments {#tbl:aligned}
```

**Alignment syntax:**
- `:---` - Left aligned (default)
- `:---:` - Center aligned
- `---:` - Right aligned

### Table Examples by Use Case

#### Model Performance Comparison
```markdown
| Algorithm     | Accuracy | Precision | Recall | F1-Score |
|---------------|:--------:|:---------:|:------:|:--------:|
| XGBoost       | 0.875    | 0.862     | 0.851  | 0.856    |
| Random Forest | 0.843    | 0.831     | 0.826  | 0.828    |
| SVM           | 0.812    | 0.805     | 0.798  | 0.801    |
| Naive Bayes   | 0.784    | 0.779     | 0.771  | 0.775    |

: Performance comparison of different machine learning algorithms {#tbl:performance}
```

#### Hyperparameter Settings
```markdown
| Parameter        | Value         | Description                    |
|:-----------------|:-------------:|:-------------------------------|
| Learning Rate    | 0.001         | Step size for optimization     |
| Batch Size       | 32            | Samples per training batch     |
| Epochs           | 100           | Number of training iterations  |
| Dropout          | 0.2           | Dropout rate for regularization|

: Hyperparameter configuration for the XGBoost model {#tbl:hyperparams}
```

#### Dataset Statistics
```markdown
| Dataset Split | Samples | Positive | Negative | Ratio    |
|:--------------|--------:|---------:|---------:|:--------:|
| Training      | 10,000  | 5,234    | 4,766    | 52:48    |
| Validation    | 2,000   | 1,045    | 955      | 52:48    |
| Test          | 2,000   | 1,038    | 962      | 52:48    |
| **Total**     | **14,000** | **7,317** | **6,683** | **52:48** |

: Dataset distribution across different splits {#tbl:dataset}
```

### Complex Tables (Grid Tables)

For tables with merged cells or complex formatting:

```markdown
+---------------+---------------+--------------------+
| Header 1      | Header 2      | Header 3           |
+===============+===============+====================+
| Row 1, Col 1  | Row 1, Col 2  | Row 1, Col 3       |
+---------------+---------------+--------------------+
| Row 2, Col 1  | Row 2, Col 2  | Row 2, Col 3       |
|               |               | Can span multiple  |
|               |               | lines              |
+---------------+---------------+--------------------+

: Complex grid table {#tbl:grid}
```

### Referencing Tables in Text

```markdown
# Single table
Table @tbl:performance shows the algorithm comparison.
→ Output: Table 1 shows the algorithm comparison.

# Multiple tables
Results are summarized in @tbl:performance and @tbl:dataset.
→ Output: Results are summarized in Tables 1 and 2.

# Parenthetical reference
The hyperparameters (@tbl:hyperparams) were optimized using grid search.
→ Output: The hyperparameters (Table 1) were optimized using grid search.
```

### Table Numbering

Tables are numbered by section (e.g., Table 1--1, Table 1--2, Table 2--1).
- First number = Section number
- Second number = Table number within that section

---

## Equations

### Inline Math

Use single `$` for inline equations:

```markdown
The equation $E = mc^2$ is Einstein's famous mass-energy equivalence.

The learning rate $\alpha$ determines convergence speed.

Accuracy is calculated as $\frac{TP + TN}{TP + TN + FP + FN}$.
```

### Display Equations (Unnumbered)

Use double `$$` for centered display equations without numbers:

```markdown
$$
\int_{-\infty}^{\infty} e^{-x^2} dx = \sqrt{\pi}
$$

$$
\nabla \cdot \mathbf{E} = \frac{\rho}{\epsilon_0}
$$
```

### Numbered Equations (for Cross-Referencing)

Add `{#eq:label}` after the equation for numbering and cross-referencing:

```markdown
$$
L = -\frac{1}{N}\sum_{i=1}^{N} [y_i \log(\hat{y}_i) + (1-y_i)\log(1-\hat{y}_i)]
$$ {#eq:loss}

# Reference it
Using @eq:loss, we minimize the binary cross-entropy.
→ Output: Using Equation 1, we minimize the binary cross-entropy.
```

### Common Equation Examples

#### Loss Functions
```markdown
# Binary Cross-Entropy Loss
$$
L_{BCE} = -\frac{1}{N}\sum_{i=1}^{N} [y_i \log(\hat{y}_i) + (1-y_i)\log(1-\hat{y}_i)]
$$ {#eq:bce}

# Mean Squared Error
$$
L_{MSE} = \frac{1}{N}\sum_{i=1}^{N} (y_i - \hat{y}_i)^2
$$ {#eq:mse}

# Categorical Cross-Entropy
$$
L_{CCE} = -\sum_{i=1}^{N}\sum_{c=1}^{C} y_{i,c} \log(\hat{y}_{i,c})
$$ {#eq:cce}
```

#### Metrics
```markdown
# Accuracy
$$
\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}
$$ {#eq:accuracy}

# Precision and Recall
$$
\text{Precision} = \frac{TP}{TP + FP}, \quad \text{Recall} = \frac{TP}{TP + FN}
$$ {#eq:precision-recall}

# F1-Score
$$
F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}
$$ {#eq:f1}
```

#### Model Equations
```markdown
# Logistic Regression
$$
P(y=1|x) = \frac{1}{1 + e^{-(\mathbf{w}^T\mathbf{x} + b)}}
$$ {#eq:logistic}

# Softmax
$$
\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}
$$ {#eq:softmax}

# Gradient Descent Update
$$
\mathbf{w}_{t+1} = \mathbf{w}_t - \alpha \nabla_{\mathbf{w}} L(\mathbf{w}_t)
$$ {#eq:gradient-descent}
```

### LaTeX Math Symbols Reference

```markdown
# Greek letters
$\alpha, \beta, \gamma, \delta, \epsilon, \theta, \lambda, \mu, \sigma, \omega$
$\Gamma, \Delta, \Theta, \Lambda, \Sigma, \Omega$

# Operators
$\sum, \prod, \int, \frac{a}{b}, \sqrt{x}, \sqrt[n]{x}$

# Relations
$=, \neq, <, >, \leq, \geq, \approx, \equiv$

# Arrows
$\rightarrow, \leftarrow, \Rightarrow, \Leftarrow, \leftrightarrow$

# Sets
$\in, \notin, \subset, \subseteq, \cup, \cap, \emptyset$

# Calculus
$\partial, \nabla, \infty, \lim, \int_{a}^{b}$

# Linear Algebra
$\mathbf{v}, \mathbf{A}, \mathbf{A}^T, \mathbf{A}^{-1}, \|\mathbf{v}\|$

# Probability
$P(A), P(A|B), \mathbb{E}[X], \text{Var}(X)$
```

### Multi-line Equations

```markdown
# Aligned equations
$$
\begin{aligned}
f(x) &= (x+1)^2 \\
     &= x^2 + 2x + 1 \\
     &= x^2 + 2x + 1
\end{aligned}
$$ {#eq:expanded}

# System of equations
$$
\begin{cases}
x + y = 5 \\
2x - y = 1
\end{cases}
$$ {#eq:system}
```

### Referencing Equations

```markdown
Equation @eq:loss defines the loss function.
→ Output: Equation 1 defines the loss function.

We use @eq:loss and @eq:accuracy for model evaluation.
→ Output: We use Equations 1 and 2 for model evaluation.

The optimization (@eq:gradient-descent) converges rapidly.
→ Output: The optimization (Equation 1) converges rapidly.
```

---

## Sections and Structure

### Section Hierarchy

```markdown
# Level 1 Heading (Section) {#sec:label1}

## Level 2 Heading (Subsection) {#sec:label2}

### Level 3 Heading (Subsubsection) {#sec:label3}

#### Level 4 Heading (Paragraph) {#sec:label4}
```

**Numbering:**
- Level 1: 1, 2, 3...
- Level 2: 1.1, 1.2, 2.1, 2.2...
- Level 3: 1.1.1, 1.1.2...
- Level 4: 1.1.1.1, 1.1.1.2...

### Unnumbered Sections

Add `{-}` or `{.unnumbered}` to make a section unnumbered:

```markdown
# Abstract {-}

# Bibliography {-}

# Appendix A {.unnumbered}
```

These sections won't be numbered but will appear in the Table of Contents.

### Complete Document Structure Example

```markdown
# Abstract {-}

This paper investigates...

# Introduction {#sec:intro}

Machine learning has revolutionized...

## Background {#sec:background}

Previous work [@smith2023] has shown...

## Motivation {#sec:motivation}

Despite these advances...

# Related Work {#sec:related}

Several approaches have been proposed...

# Methods {#sec:methods}

Our approach consists of...

## Data Collection {#sec:data}

We collected data from...

## Model Architecture {#sec:architecture}

The model architecture (@fig:architecture) consists of...

## Training Procedure {#sec:training}

Training was performed using @eq:loss...

# Results {#sec:results}

## Performance Metrics {#sec:metrics}

Table @tbl:performance shows...

## Comparison with Baselines {#sec:comparison}

Our method outperforms...

# Discussion {#sec:discussion}

The results in @sec:results demonstrate...

# Conclusion {#sec:conclusion}

We presented a novel approach...

# Bibliography {-}
```

### Referencing Sections

```markdown
As discussed in @sec:intro, machine learning...
→ Output: As discussed in Section 1, machine learning...

The methodology (@sec:methods) is based on...
→ Output: The methodology (Section 3) is based on...

Sections @sec:results and @sec:discussion analyze...
→ Output: Sections 4 and 5 analyze...
```

---

## Text Formatting

### Basic Text Styles

```markdown
*italic text* or _italic text_
→ Output: italic text

**bold text** or __bold text__
→ Output: bold text

***bold and italic*** or ___bold and italic___
→ Output: bold and italic

`inline code` or `variable_name`
→ Output: inline code (monospace font)

~~strikethrough text~~
→ Output: ~~strikethrough text~~

[Hyperlink text](https://example.com)
→ Output: Hyperlink text (clickable link)
```

### Lists

#### Unordered Lists
```markdown
- First item
- Second item
  - Nested item 2.1
  - Nested item 2.2
    - Deeply nested item
- Third item

# Alternative bullets
* Item with asterisk
+ Item with plus
```

#### Ordered Lists
```markdown
1. First item
2. Second item
   1. Nested item 2.1
   2. Nested item 2.2
3. Third item

# Auto-numbering (all items marked as 1.)
1. First item
1. Second item (will be numbered 2)
1. Third item (will be numbered 3)
```

#### Mixed Lists
```markdown
1. Ordered item 1
   - Unordered sub-item
   - Another sub-item
2. Ordered item 2
   1. Ordered sub-item
   2. Another ordered sub-item
```

#### Definition Lists
```markdown
Term 1
:   Definition of term 1

Term 2
:   Definition of term 2.
    Can span multiple lines with proper indentation.

Machine Learning
:   A field of artificial intelligence that enables systems to learn from data.

Deep Learning
:   A subset of machine learning using neural networks with multiple layers.
```

### Code Blocks

#### Inline Code
```markdown
Use the `fit()` method to train the model.
The variable `learning_rate` controls convergence.
```

#### Code Blocks without Syntax Highlighting
````markdown
```
def hello():
    print("Hello, World!")
```
````

#### Code Blocks with Syntax Highlighting

Python:
````markdown
```python
def train_model(X_train, y_train):
    model = XGBClassifier(
        learning_rate=0.001,
        max_depth=5,
        n_estimators=100
    )
    model.fit(X_train, y_train)
    return model
```
````

R:
````markdown
```r
model <- train(
  target ~ .,
  data = train_data,
  method = "xgbTree"
)
```
````

SQL:
````markdown
```sql
SELECT team_name, AVG(score) as avg_score
FROM games
WHERE season = 2023
GROUP BY team_name
ORDER BY avg_score DESC;
```
````

Bash:
````markdown
```bash
#!/bin/bash
cd doc
./build.sh
echo "Build complete!"
```
````

### Block Quotes

```markdown
> This is a block quote.
> It can span multiple lines.

> **Important:** Block quotes can contain formatting.
>
> They can also contain multiple paragraphs.

> Nested quotes:
> > This is a nested quote.
> > > And this is double-nested.
```

### Footnotes

```markdown
This statement needs a citation.[^1]

Another statement with explanation.[^note]

[^1]: This is the footnote content for reference 1.

[^note]: This is a named footnote. You can use any identifier.
    Footnotes can span multiple lines with proper indentation.
```

### Horizontal Rules

```markdown
---

or

***

or

___
```

All produce a horizontal line separator.

### Line Breaks

```markdown
# Soft break (two spaces at end of line)
Line one  
Line two

# Hard break (backslash)
Line one\
Line two

# Paragraph break (blank line)
Paragraph one

Paragraph two
```

### Special Characters and Escaping

```markdown
# Escape special characters with backslash
\* Not a bullet point
\# Not a heading
\[ Not a link

# HTML entities
&copy; → ©
&reg; → ®
&trade; → ™
&mdash; → —
&ndash; → –
```

---

## Advanced Features

### Div Blocks (Custom Containers)

```markdown
::: {.warning}
⚠️ **Warning:** This is an important warning message.
:::

::: {.note}
📝 **Note:** Additional information goes here.
:::

::: {#custom-id .custom-class}
Content with custom ID and class for CSS styling.
:::
```

### Span (Inline Formatting)

```markdown
This is [important text]{.highlight}.

This is [text with custom class]{.custom-class}.

This is [text with ID]{#custom-id}.
```

### Page Breaks

```markdown
Content before page break.

\newpage

Content after page break (starts on new page).
```

### Comments (Not Rendered)

```markdown
<!-- This is a comment and won't appear in the PDF -->

<!--
Multi-line comment
won't appear either
-->
```

### Raw LaTeX

You can insert raw LaTeX commands:

```markdown
\clearpage   % Clear page and flush floats

\vspace{1cm} % Add vertical space

\noindent    % Prevent paragraph indentation

\textcolor{red}{This text is red}
```

### Subscripts and Superscripts

```markdown
# In regular text
H~2~O renders as H₂O
E = mc^2^ renders as E = mc²

# In math mode (better for complex expressions)
$H_2O$
$E = mc^2$
$x^{2^{n}}$ (nested superscripts)
$x_{i,j}$ (subscripts with comma)
```

---

## Configuration

### Main Configuration File: `defaults.yaml`

```yaml
from: markdown
to: pdf
pdf-engine: xelatex
template: template.tex
filters:
  - pandoc-crossref   # Must come before citeproc!
  - citeproc
bibliography: references.bib
csl: apa.csl
standalone: true
output-file: report.pdf
number-sections: true
resource-path:
  - src  # Helps find images in src directory

metadata:
  # Document Information
  doc-title: "Comparing Machine Learning Methods for Sports"
  doc-subtitle: "A Case Study on NCAA Basketball"
  doc-semester: "HS25"
  doc-course: "AICOMP"
  doc-authors: "Luca Kyburz, David Schurtenberger"
  
  # Cross-reference Configuration
  figureTitle: "Figure"
  tableTitle: "Table"
  figPrefix: "Figure"
  eqnPrefix: "Equation"
  tblPrefix: "Table"
  secPrefix: "Section"
  
  # Document Features
  lof: true   # List of Figures
  lot: true   # List of Tables

variables:
  fontsize: 11pt
  tables: true
  secnumdepth: 4
```

### Customization Options

#### Change Citation Style

Replace `apa.csl` with another CSL file:
- `chicago.csl` - Chicago style
- `ieee.csl` - IEEE style
- `vancouver.csl` - Vancouver style
- `harvard.csl` - Harvard style

Download CSL files from: https://www.zotero.org/styles

```yaml
csl: chicago.csl  # or ieee.csl, etc.
```

#### Disable List of Figures/Tables

```yaml
metadata:
  lof: false  # Disable List of Figures
  lot: false  # Disable List of Tables
```

#### Change Cross-Reference Format

```yaml
metadata:
  figPrefix: "Fig."        # Use "Fig." instead of "Figure"
  tblPrefix: "Tab."        # Use "Tab." instead of "Table"
  eqnPrefix: "Eq."         # Use "Eq." instead of "Equation"
  secPrefix: "Sec."        # Use "Sec." instead of "Section"
```

#### Adjust Table of Contents Depth

```yaml
metadata:
  toc: true
  toc-depth: 2  # Only show level 1 and 2 headings
```

#### Change Font Size

```yaml
variables:
  fontsize: 10pt  # or 11pt, 12pt
```

---

## Troubleshooting

### Common Issues and Solutions

#### Problem: Cross-references show "??"

**Cause:** Label not defined or typo in label name

**Solutions:**
```markdown
# ✅ Correct
![Caption](image.png){#fig:model}
Reference: @fig:model

# ❌ Wrong - typo in reference
![Caption](image.png){#fig:model}
Reference: @fig:mdoel  ← typo!

# ❌ Wrong - missing label
![Caption](image.png)
Reference: @fig:model  ← no label defined!
```

#### Problem: Citations not appearing or showing as [@smith2023]

**Cause:** Citation key not in `references.bib` or citeproc not enabled

**Solutions:**
1. Check the citation key exists in `references.bib`
2. Verify `citeproc` is in the filters list in `defaults.yaml`
3. Ensure `bibliography: references.bib` is set

#### Problem: Images not showing in PDF

**Cause:** Incorrect file path or missing image

**Solutions:**
```markdown
# ✅ Correct - relative to src/ directory
![Caption](./images/model.png){#fig:model}

# ❌ Wrong - absolute path won't work in Docker
![Caption](C:/Users/me/images/model.png){#fig:model}

# Check image exists
ls src/images/model.png
```

#### Problem: "Figure Figure 1" duplication

**Cause:** Writing "Figure @fig:label" when pandoc-crossref adds "Figure" automatically

**Solution:**
```markdown
# ❌ Wrong
As shown in Figure @fig:model...
→ Output: As shown in Figure Figure 1...

# ✅ Correct
As shown in @fig:model...
→ Output: As shown in Figure 1...
```

#### Problem: Table not numbered correctly

**Cause:** Missing caption with ID

**Solution:**
```markdown
# ❌ Wrong - no caption
| Col1 | Col2 |
|------|------|
| A    | B    |

# ✅ Correct - with caption and ID
| Col1 | Col2 |
|------|------|
| A    | B    |

: Table caption {#tbl:data}
```

#### Problem: Build fails with "Permission denied"

**Cause:** PDF file is open in a viewer

**Solution:**
```bash
# Close the PDF file, then:
cd doc
rm -f _build/report.pdf
./build.sh
```

#### Problem: Equation numbering not working

**Cause:** Missing `{#eq:label}` or incorrect placement

**Solution:**
```markdown
# ❌ Wrong - no label
$$E = mc^2$$

# ❌ Wrong - label inside equation
$$E = mc^2 {#eq:einstein}$$

# ✅ Correct - label after equation block
$$E = mc^2$$ {#eq:einstein}
```

#### Problem: Bibliography not appearing

**Cause:** No bibliography section or no citations used

**Solution:**
```markdown
# Add at end of document
# Bibliography {-}

# Make sure you've cited at least one reference
Recent work [@smith2023] shows...
```

#### Problem: Math symbols not rendering

**Cause:** Special LaTeX characters need escaping or proper syntax

**Solution:**
```markdown
# ✅ Correct
$\alpha, \beta, \gamma$  ← Use backslash for Greek letters
$x^2 + y^2 = z^2$        ← Use ^ for superscripts
$x_i$                    ← Use _ for subscripts
$\frac{a}{b}$            ← Use \frac for fractions

# ❌ Wrong
$α, β, γ$                ← Direct Unicode may not work
```

---

## Complete Working Example

Here's a complete `src/report.md` example demonstrating all features:

```markdown
# Abstract {-}

This paper compares machine learning algorithms for NCAA basketball game 
prediction. We evaluate XGBoost, Random Forest, and SVM on historical game 
data [@ncaa2023; @smith2023].

# Introduction {#sec:intro}

Machine learning has revolutionized sports analytics [@johnson2022, ch. 3]. 
As shown in @fig:architecture, our approach builds upon existing work to 
predict game outcomes with high accuracy.

![Proposed model architecture overview](./images/model.png){#fig:architecture width=70%}

According to @smith2023 [p. 42], feature engineering is critical for model 
performance. Our work extends these insights to basketball analytics.

## Research Questions {#sec:questions}

We investigate the following questions:

1. Which algorithm performs best for game prediction?
2. How do different features impact prediction accuracy?
3. Can we achieve real-time predictions?

## Contributions {#sec:contributions}

Our main contributions are:

- Novel feature engineering approach
- Comprehensive algorithm comparison
- Real-time prediction system

# Related Work {#sec:related}

Previous studies [@brown2021; @davis2020] have explored various approaches 
to sports prediction. However, most focus on single algorithms rather than 
comprehensive comparisons.

# Methods {#sec:methods}

## Dataset {#sec:data}

We collected data from NCAA games spanning 2010-2023. Table @tbl:dataset 
summarizes the dataset statistics.

| Split      | Games   | Teams | Features |
|:-----------|--------:|------:|---------:|
| Training   | 10,000  | 350   | 42       |
| Validation | 2,000   | 350   | 42       |
| Test       | 2,000   | 350   | 42       |

: Dataset statistics across different splits {#tbl:dataset}

## Feature Engineering {#sec:features}

We engineered 42 features including:

- **Historical performance**: Win rate, scoring average, ELO rating
- **Recent form**: Last 5 game outcomes, scoring trends
- **Matchup statistics**: Head-to-head record, venue advantage

The ELO rating is calculated using @eq:elo:

$$
ELO_{new} = ELO_{old} + K \cdot (S - E)
$$ {#eq:elo}

where $K$ is the K-factor, $S$ is the actual score (1 for win, 0 for loss), 
and $E$ is the expected score.

## Model Architecture {#sec:architecture}

We evaluated three algorithms. The hyperparameters (see @tbl:hyperparams) 
were optimized using 5-fold cross-validation.

| Parameter      | XGBoost | Random Forest | SVM     |
|:---------------|:--------|:--------------|:--------|
| Learning Rate  | 0.001   | N/A           | N/A     |
| Max Depth      | 5       | 10            | N/A     |
| N Estimators   | 100     | 200           | N/A     |
| Kernel         | N/A     | N/A           | RBF     |
| C              | N/A     | N/A           | 1.0     |

: Optimized hyperparameters for each algorithm {#tbl:hyperparams}

## Training Procedure {#sec:training}

Models were trained using binary cross-entropy loss (@eq:loss):

$$
L = -\frac{1}{N}\sum_{i=1}^{N} [y_i \log(\hat{y}_i) + (1-y_i)\log(1-\hat{y}_i)]
$$ {#eq:loss}

where $N$ is the number of samples, $y_i$ is the true label, and 
$\hat{y}_i$ is the predicted probability.

Training was performed on an NVIDIA RTX 3090 GPU[^gpu] using early stopping 
with patience of 10 epochs.

[^gpu]: Training time was approximately 2 hours for all models combined.

# Results {#sec:results}

## Performance Metrics {#sec:metrics}

Table @tbl:performance presents the main results. XGBoost achieved the 
highest accuracy (0.875), outperforming both Random Forest and SVM.

| Algorithm     | Accuracy | Precision | Recall | F1-Score |
|:--------------|:--------:|:---------:|:------:|:--------:|
| **XGBoost**   | **0.875**| **0.862** |**0.851**|**0.856**|
| Random Forest | 0.843    | 0.831     | 0.826  | 0.828    |
| SVM           | 0.812    | 0.805     | 0.798  | 0.801    |

: Performance comparison of different algorithms on test set {#tbl:performance}

Performance metrics were calculated using standard definitions:

$$
\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}
$$ {#eq:accuracy}

$$
F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}
$$ {#eq:f1}

## Analysis {#sec:analysis}

The results demonstrate several key findings:

1. **Algorithm performance**: As shown in @tbl:performance, XGBoost 
   consistently outperformed other algorithms across all metrics.

2. **Feature importance**: ELO rating (@eq:elo) was the most important 
   feature, contributing 35% to model predictions.

3. **Training stability**: Using @eq:loss, all models converged within 
   50 epochs without overfitting.

![Training loss over epochs for all three algorithms](./images/loss_curve.png){#fig:loss width=80%}

Figure @fig:loss shows the training progression. XGBoost converged fastest, 
reaching optimal loss in 30 epochs.

# Discussion {#sec:discussion}

Our findings confirm previous observations by @smith2023 [pp. 45-48] that 
gradient boosting methods excel at structured data prediction. However, we 
extend this work by demonstrating superior performance specifically for 
basketball analytics.

The architecture (@fig:architecture) and loss function (@eq:loss) proved 
effective across all tested scenarios. As noted in @sec:results, the 
combination of historical and recent features provided robust predictions.

# Conclusion {#sec:conclusion}

We presented a comprehensive comparison of machine learning algorithms for 
NCAA basketball prediction. Our experiments (@sec:methods) demonstrated that 
XGBoost achieves state-of-the-art performance (Table @tbl:performance).

Future work will explore:

- Deep learning architectures
- Transfer learning from professional leagues
- Real-time prediction systems

The complete methodology is described in Sections @sec:methods through 
@sec:results, with reproducible code available online.

# Bibliography {-}
```

---

## Quick Reference Card

### Most Common Syntax

```markdown
# Citations
[@smith2023]                    → (Smith & Doe, 2023)
[@smith2023, p. 42]             → (Smith & Doe, 2023, p. 42)
@smith2023                      → Smith and Doe (2023)

# Cross-references
@fig:label                      → Figure 1
@tbl:label                      → Table 1
@eq:label                       → Equation 1
@sec:label                      → Section 1

# Figures
![Caption](./images/img.png){#fig:label width=70%}

# Tables
| Col1 | Col2 |
|------|------|
| A    | B    |

: Caption {#tbl:label}

# Equations
$$E = mc^2$$ {#eq:label}

# Sections
# Heading {#sec:label}

# Bibliography
# Bibliography {-}
```

---

## Additional Resources

### Pandoc Documentation
- Main docs: https://pandoc.org/MANUAL.html
- Markdown syntax: https://pandoc.org/MANUAL.html#pandocs-markdown

### Pandoc-Crossref
- Documentation: https://lierdakil.github.io/pandoc-crossref/

### Citation Styles
- CSL repository: https://www.zotero.org/styles
- CSL documentation: https://citationstyles.org/

### LaTeX Math
- Math symbols: https://www.overleaf.com/learn/latex/List_of_Greek_letters_and_math_symbols
- Equation formatting: https://www.overleaf.com/learn/latex/Mathematical_expressions

---

## File Structure

```
doc/
├── build.sh                    # Build script (Linux/macOS/Git Bash)
├── build.bat                   # Build script (Windows)
├── defaults.yaml               # Pandoc configuration
├── template.tex                # LaTeX template
├── references.bib              # Bibliography database
├── apa.csl                     # Citation style (APA)
├── README.md                   # This file
├── src/
│   ├── report.md              # Your markdown content
│   └── images/                # Image files
│       ├── model.png
│       └── ...
└── _build/
    └── report.pdf             # Generated output
```
