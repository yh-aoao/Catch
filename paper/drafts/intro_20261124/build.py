from pathlib import Path
from docx import Document
from docx.shared import Cm, Pt, RGBColor
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
R=Path(__file__).parent
paras=[
'接住运动中的物体，以及将手中的物体抛向指定位置，是两类具有代表性的动态操作任务。前者要求机器人根据目标运动及时调整接触位置与时机，并在接触后稳定控制物体；后者则要求机器人协调自身运动与释放动作，使物体到达指定区域[1–3]。同时具备接取与投掷能力，使机器人既能接收外部传来的物体，也能主动将物体传出，为双向物体传递提供基础。例如，在人与机器人的交互中，机器人可以接住人抛来的物体，再根据任务要求将其抛回或投向指定目标。面向这一应用方向，机器人需要在同一平台上掌握动态接取与主动投掷技能，并能够根据任务要求执行相应操作。',
'实现上述能力，需要解决动态响应、技能整合与移动协同三个方面的问题。首先，动态接取具有严格的时间约束：机器人必须在目标经过时到达合适位置，并协调接触时机、手部闭合与接触后的物体稳定[1,3]。不同运动情境进一步增加了控制难度，直接抛来、沿桌面滚落和经地面反弹的物体具有不同的运动演化与接触过程，对拦截位置和响应时机提出了不同要求。其次，接取需要适应外部物体的运动，投掷则需要主动生成合适的释放状态[1,2]。将这些技能整合到同一策略中，需要在共享感知与时序表征的同时，保留各项技能所需的控制差异。最后，移动能力增加了操作位置的选择：固定底座机械臂的接取范围受到其工作空间限制，移动平台则允许机器人调整自身位置，为接取原本不可达的目标提供可能；与此同时，底盘运动也增加了与机械臂、灵巧手协同控制的难度[3]。',
'已有研究探索了固定底座与移动操作平台上的动态物体接取。Kim等通过学习物体运动与机械臂动作，联合处理飞行轨迹预测、可行接取构型选择与机械臂运动生成[1]。Catch It进一步展示了移动底盘、机械臂与灵巧手协同接住飞行物体的可行性，并提出先学习跟踪、再学习接取的两阶段强化学习流程[3]。另一方面，TossingBot展示了从视觉观测中学习抓取与投掷控制参数、将物体投向指定容器的能力[2]。在这些研究基础上，本文关注如何将直接抛来、桌面滚落和地面反弹三种情境下的接取能力，与主动投框能力整合到同一个视觉策略中。',
'为此，本文采用分任务教师训练与多教师视觉策略蒸馏相结合的框架。策略蒸馏已被用于将多个任务的专用策略整合到单个网络中[4]，本文将这一思路用于移动机器人的动态接取与主动投掷。首先，通过两阶段强化学习分别训练三种接取任务与主动投框任务的专用教师。接取教师先学习目标跟踪，再学习包含手部抓握的完整接取动作；投框教师先在固定底盘条件下学习面向不同目标距离的投掷，再学习自主移动、选择停靠位置并完成投掷。各任务在第二阶段均继续优化第一阶段的策略参数。随后，将教师的控制能力蒸馏到共享视觉感知与时序记忆的学生策略中。学生根据连续视觉观测、机器人自身状态与任务指令，通过接取和投掷两个动作分支生成控制命令，在共享表征的同时保留两类操作的控制差异。',
'针对动态接取中的运动信息提取，学生策略结合当前位置辅助监督与未来物体位移预测。已有视觉操作研究通过预测物体三维位置，为视觉表征学习提供额外监督[5]。本文采用训练期当前位置辅助监督，其预测输出不直接输入动作分支，而是通过辅助损失参与共享表征的学习。在此基础上，未来物体位移预测进一步融合到接取动作生成中，为策略提供目标运动趋势信息。两者分别从当前位置感知与未来运动表征两个方面，为动态接取策略的学习提供支持。',
'本文将在仿真环境中，通过三种接取情境与主动投框任务评估学生策略的多技能表现，并通过消融实验分析当前位置辅助监督与未来位移预测的作用。在真实移动操作机器人上，将进一步评估所学策略的接取与投掷能力及其从仿真迁移到真实环境的表现，并通过连续接取与投掷实验考察不同技能的衔接能力，探索其在人机物体传递中的应用。'
]
refs=[
('S. Kim, A. Shukla, and A. Billard, “Catching Objects in Flight,” IEEE Transactions on Robotics, vol. 30, no. 5, pp. 1049–1065, 2014. DOI: 10.1109/TRO.2014.2316022.','https://doi.org/10.1109/TRO.2014.2316022'),
('A. Zeng, S. Song, J. Lee, A. Rodriguez, and T. Funkhouser, “TossingBot: Learning to Throw Arbitrary Objects with Residual Physics,” Robotics: Science and Systems, 2019.','https://gfx.cs.princeton.edu/pubs/Zeng_2019_TLT/index.php'),
('Y. Zhang, T. Liang, Z. Chen, Y. Ze, and H. Xu, “Catch It! Learning to Catch in Flight with Mobile Dexterous Hands,” arXiv:2409.10319, 2024.','https://arxiv.org/abs/2409.10319'),
('A. A. Rusu, S. Gomez Colmenarejo, C. Gulcehre, G. Desjardins, J. Kirkpatrick, R. Pascanu, V. Mnih, K. Kavukcuoglu, and R. Hadsell, “Policy Distillation,” arXiv:1511.06295, 2015.','https://arxiv.org/abs/1511.06295'),
('Y. Zhu, Z. Wang, J. Merel, A. Rusu, T. Erez, S. Cabi, S. Tunyasuvunakool, J. Kramár, R. Hadsell, N. de Freitas, and N. Heess, “Reinforcement and Imitation Learning for Diverse Visuomotor Skills,” Robotics: Science and Systems, 2018.','https://arxiv.org/abs/1802.09564')]
d=Document();s=d.sections[0];s.page_width=Cm(21);s.page_height=Cm(29.7);s.top_margin=Cm(2.3);s.bottom_margin=Cm(2.3);s.left_margin=Cm(2.5);s.right_margin=Cm(2.5)
for name in ['Normal','Title','Heading 1']:
 st=d.styles[name];st.font.name='Times New Roman';st.font.color.rgb=RGBColor(0,0,0);st._element.get_or_add_rPr().rFonts.set(qn('w:eastAsia'),'Noto Serif CJK SC')
st=d.styles['Normal'];st.font.size=Pt(11);pf=st.paragraph_format;pf.line_spacing=1.3;pf.space_after=Pt(7);pf.first_line_indent=Pt(22);pf.widow_control=True
st=d.styles['Title'];st.font.size=Pt(18);st.paragraph_format.space_after=Pt(14)
for style in d.styles:
 for border in list(style.element.iter(qn('w:pBdr'))): border.getparent().remove(border)
d.styles['Title'].paragraph_format.first_line_indent=Pt(0)
d.styles['Heading 1'].paragraph_format.first_line_indent=Pt(0)
d.add_paragraph('移动机器人动态接取与投掷论文引言',style='Title')
for text in paras:d.add_paragraph(text)
d.add_page_break();d.add_heading('参考文献',level=1)
from docx.opc.constants import RELATIONSHIP_TYPE as RT
for i,(text,url) in enumerate(refs,1):
 p=d.add_paragraph();p.paragraph_format.first_line_indent=Pt(-17);p.paragraph_format.left_indent=Pt(17);p.paragraph_format.space_after=Pt(12);p.paragraph_format.line_spacing=1.2;p.add_run(f'[{i}] {text} ')
 h=OxmlElement('w:hyperlink');h.set(qn('r:id'),p.part.relate_to(url,RT.HYPERLINK,is_external=True));r=OxmlElement('w:r');pr=OxmlElement('w:rPr');col=OxmlElement('w:color');col.set(qn('w:val'),'245B88');pr.append(col);r.append(pr);t=OxmlElement('w:t');t.text=url;r.append(t);h.append(r);p._p.append(h)
d.core_properties.title='移动机器人动态接取与投掷论文引言';d.core_properties.author='';d.core_properties.last_modified_by=''
d.save(R/'Introduction_中文完善及引用版.docx')
(R/'intro.txt').write_text('\n\n'.join(paras)+'\n\n参考文献\n'+ '\n\n'.join(f'[{i}] {t}\n{u}' for i,(t,u) in enumerate(refs,1)))
(R/'citation_notes.md').write_text('检索日期：2026-11-24。定向引言引用核查，不是系统综述。检索来源：arXiv、IEEE Xplore、作者项目页。\n\n[1] IEEE原始摘要：飞行轨迹预测、可行接取构型及毫秒级运动规划。\n[2] 作者页及论文：指定容器投掷、初始条件和物体动力学，不能用作人机双向传递已实现的证据。\n[3] arXiv原文Introduction及Two-Stage RL：移动全身协同、固定基座可达性、跟踪到接取训练。采用2024预印本条目，不混用2025会议元数据。\n[4] arXiv摘要：多专用策略整合为单一策略；Atari领域，不能据此声称本机器人技能整合已有性能保证。\n[5] 原文III-B-2第4项：CNN输出物体三维位置的辅助回归、额外监督。它是相近设计的可核查先例，尚未确认是用户实际迁移所依据的论文。\n\n本项目方法与未来预测融合不引用外文作为自身性能证明。尚未确认仿真消融、真机与连续传递结果完成情况，实验末段采用将来时，待实际结果后改写。\n')
