引言及参考文献导入说明

1. 在Overleaf上传intro.tex和intro_refs.bib。
2. 将原引言替换为\input{intro}。如宿主文件已经有\section{Introduction}，删除intro.tex里的同名section和重复label。
3. 若使用BibTeX，将intro_refs追加到现有\bibliography{...}中，例如\bibliography{references,intro_refs}，保留原bibliographystyle。若相同文献已有条目，统一citation key并去重。
4. 若使用biblatex，在导言区添加\addbibresource{intro_refs.bib}，沿用项目原有的\printbibliography；两套文献系统不要混用。
5. 正文为用户当前中文稿。项目需已有中文支持；如有需要，用XeLaTeX并在导言区加载ctex。正式英文RSS稿需翻译，不要将本中文稿当英文终稿。
6. 本包没有访问项目源文件，未核查其主文件、标签、文献冲突或编译。文件已经完成12篇文献citation-key对应检查。
7. 真机、消融及连续接抛按计划语气；结果完成后改写。Zhu2018是相近辅助设计先例，尚未确认是实际迁移来源。
