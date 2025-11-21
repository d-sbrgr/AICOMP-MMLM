# Report Documentation

This guide explains how to use Markdown syntax in `src/report.md` to create a professional PDF report with proper citations, images, tables, and cross-references.

## Building the Report

### On Linux/macOS or Windows (Git Bash)
```bash
./build.sh
```

### On Windows (Command Prompt)
```bat
build.bat
```

The script will:
1. Build the Docker image `pandoc-report-builder` (if not already built)
2. Create the `_build` directory
3. Generate `_build/report.pdf` from `src/report.md`

---

## Markdown Syntax Guide

### 1. Citations and Bibliography

#### Adding References
Add your references to `references.bib` in BibTeX format:

```bibtex
@article{author2023,
  author = {Smith, John and Doe, Jane},
  title = {A Great Research Paper},
  journal = {Journal of Machine Learning},
  year = {2023},
  volume = {42},
  pages = {123--145}
}

@book{author2022,
  author = {Johnson, Alice},
  title = {Introduction to Data Science},
  publisher = {Academic Press},
  year = {2022}
}

@inproceedings{author2021,
  author = {Brown, Bob},
  title = {Deep Learning for Sports Analytics},
  booktitle = {Proceedings of the International Conference on AI},
  year = {2021},
  pages = {45--52}
}
```

#### Citing in Text
Use the `@` symbol followed by the citation key:

```markdown
According to recent research [@author2023], machine learning has shown...

Multiple citations can be combined [@author2023; @author2022; @author2021].

You can also add page numbers or prefixes [@author2023, pp. 125-130].

For in-text citations: @author2023 showed that...
```

**Output examples:**
- `[@author2023]` → (Smith & Doe, 2023)
- `@author2023` → Smith and Doe (2023)
- `[@author2023; @author2022]` → (Smith & Doe, 2023; Johnson, 2022)

#### Bibliography Section
The bibliography is automatically generated. In your `report.md`, add:

```markdown
# Bibliography

[bibliography]
```

This will insert all cited references in APA format (as specified in `defaults.yaml`).

---

### 2. Images and Figures

#### Basic Image
```markdown
![Caption text](path/to/image.png)
```

#### Image with ID (for cross-referencing)
```markdown
![Model architecture overview](../assets/architecture.png){#fig:architecture}
```

#### Image with Custom Size
```markdown
![Training results](../assets/results.png){#fig:results width=80%}
```

#### Multiple Images Side-by-Side
```markdown
::: {#fig:comparison}
![Model A](../assets/model_a.png){width=45%}
![Model B](../assets/model_b.png){width=45%}

Comparison of Model A and Model B performance.
:::
```

#### Referencing Figures in Text
```markdown
As shown in Figure @fig:architecture, our model consists of...

The results (see @fig:results) demonstrate that...

Figures @fig:architecture and @fig:results show...
```

---

### 3. Tables

#### Basic Table
```markdown
| Algorithm | Accuracy | F1-Score |
|-----------|----------|----------|
| XGBoost   | 0.85     | 0.83     |
| Random Forest | 0.82  | 0.80     |
| SVM       | 0.78     | 0.76     |

: Comparison of model performance {#tbl:performance}
```

#### Table with Alignment
```markdown
| Left-aligned | Center-aligned | Right-aligned |
|:-------------|:--------------:|--------------:|
| Text         | Text           | 123.45        |
| More text    | More text      | 678.90        |

: Table caption here {#tbl:example}
```

#### Complex Table (using Pandoc's grid tables)
```markdown
+---------------+---------------+--------------------+
| Header 1      | Header 2      | Header 3           |
+===============+===============+====================+
| Row 1, Col 1  | Row 1, Col 2  | Row 1, Col 3       |
+---------------+---------------+--------------------+
| Row 2, Col 1  | Row 2, Col 2  | Row 2, Col 3       |
+---------------+---------------+--------------------+

: Complex table example {#tbl:complex}
```

#### Referencing Tables in Text
```markdown
Table @tbl:performance shows the comparison between different algorithms.

As demonstrated in @tbl:performance and @tbl:example, ...
```

---

### 4. Cross-References

#### Sections
```markdown
# Introduction {#sec:intro}

## Background {#sec:background}

# Methods {#sec:methods}
```

Reference them with:
```markdown
As discussed in Section @sec:intro, we aim to...

See @sec:methods for details.
```

#### Equations
```markdown
The loss function is defined as:

$$L = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$ {#eq:mse}

Where $n$ is the number of samples.
```

Reference with:
```markdown
Equation @eq:mse shows the mean squared error calculation.
```

---

### 5. Text Formatting

#### Basic Formatting
```markdown
*italic text* or _italic text_

**bold text** or __bold text__

***bold and italic*** or ___bold and italic___

`inline code`

~~strikethrough~~
```

#### Lists

**Unordered:**
```markdown
- Item 1
- Item 2
  - Nested item 2.1
  - Nested item 2.2
- Item 3
```

**Ordered:**
```markdown
1. First item
2. Second item
   1. Nested item
   2. Another nested item
3. Third item
```

#### Code Blocks

**Without language:**
```markdown
```
Code here
```
```

**With syntax highlighting:**
````markdown
```python
def train_model(X, y):
    model = XGBClassifier()
    model.fit(X, y)
    return model
```
````

#### Block Quotes
```markdown
> This is a block quote.
> It can span multiple lines.
>
> And multiple paragraphs.
```

---

### 6. Advanced Features

#### Footnotes
```markdown
This is some text with a footnote.[^1]

[^1]: This is the footnote content.
```

#### Definition Lists
```markdown
Term 1
:   Definition of term 1

Term 2
:   Definition of term 2
    Can span multiple lines
```

#### Line Breaks
```markdown
End a line with two spaces  
to create a line break.

Or use a backslash\
for the same effect.
```

#### Horizontal Rule
```markdown
---

or

***

or

___
```

#### Math Equations

**Inline math:**
```markdown
The equation $E = mc^2$ is famous.
```

**Display math:**
```markdown
$$
\int_{-\infty}^{\infty} e^{-x^2} dx = \sqrt{\pi}
$$
```

---

### 7. Special Pandoc Features

#### Div Blocks
```markdown
::: {.warning}
This is a warning box.
:::

::: {#custom-div .myclass}
Content with custom ID and class.
:::
```

#### Span
```markdown
This is [important text]{.highlight}.
```

#### Page Breaks
```markdown
\newpage
```

---

## Complete Example

Here's a complete example combining multiple features:

```markdown
# Introduction {#sec:intro}

Recent advances in machine learning [@smith2023; @doe2022] have shown 
promising results in sports analytics. As shown in Figure @fig:model, 
our approach builds upon these foundations.

![Proposed model architecture](../assets/model.png){#fig:model width=70%}

## Methodology {#sec:method}

We evaluated three algorithms as summarized in Table @tbl:results:

| Algorithm     | Accuracy | Precision | Recall |
|---------------|----------|-----------|--------|
| XGBoost       | 0.875    | 0.862     | 0.851  |
| Random Forest | 0.843    | 0.831     | 0.826  |
| SVM           | 0.812    | 0.805     | 0.798  |

: Performance comparison of different models {#tbl:results}

The loss function used is:

$$L = -\frac{1}{N}\sum_{i=1}^{N} [y_i \log(\hat{y}_i) + (1-y_i)\log(1-\hat{y}_i)]$$ {#eq:loss}

where $N$ is the number of samples and $\hat{y}_i$ represents the predicted probability.

## Results {#sec:results}

As demonstrated in @sec:method, our approach using Equation @eq:loss 
achieved the results shown in @tbl:results. The model architecture 
(@fig:model) was trained on NCAA basketball data.[^1]

[^1]: Data sourced from the Kaggle March Madness competition.

# Bibliography

[bibliography]
```

---

## Tips and Best Practices

1. **Image Paths**: Use relative paths from the `doc` directory (e.g., `../assets/image.png`)
2. **IDs**: Use descriptive IDs for cross-references (e.g., `#fig:architecture`, `#tbl:results`)
3. **Citations**: Always add citation keys to `references.bib` before using them
4. **Tables**: For complex tables, use grid tables or consider generating them from data
5. **Figures**: Keep figure widths at 70-80% for good readability
6. **Sections**: Use meaningful section IDs for easier cross-referencing
7. **Math**: Use `$$` for display equations and `$` for inline math
8. **Build Often**: Run the build script frequently to catch errors early

---

## Troubleshooting

### Common Issues

**Missing citations:**
- Ensure the citation key exists in `references.bib`
- Check that `citeproc: true` is set in `defaults.yaml`

**Broken cross-references:**
- Verify that the ID you're referencing exists
- Use the correct prefix (`@fig:`, `@tbl:`, `@sec:`, `@eq:`)

**Images not showing:**
- Check the file path is correct relative to the `doc` directory
- Ensure the image file exists

**Build failures:**
- Check for unclosed code blocks or brackets
- Verify YAML metadata is valid
- Look for special characters that need escaping

---

## Configuration Files

### `defaults.yaml`
Contains Pandoc settings including:
- PDF engine (xelatex)
- Citation style (APA via `apa.csl`)
- Bibliography file (`references.bib`)
- Template (`template.tex`)
- Metadata (title, authors, course info)

### `references.bib`
BibTeX file containing all your references.

### `template.tex`
LaTeX template for PDF formatting.

---

## Resources

- [Pandoc Manual](https://pandoc.org/MANUAL.html)
- [Pandoc Markdown](https://pandoc.org/MANUAL.html#pandocs-markdown)
- [Citation Syntax](https://pandoc.org/MANUAL.html#citation-syntax)
- [Cross-referencing](https://pandoc.org/MANUAL.html#extension-link_attributes)
- [BibTeX Format](http://www.bibtex.org/Format/)

