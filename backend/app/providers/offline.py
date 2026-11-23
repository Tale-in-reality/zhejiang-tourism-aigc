from __future__ import annotations

import hashlib
from typing import Any

from .llm import LLMProvider, LLMResult, plan_narration_budget
from ..services import knowledge as kb


CITY_KNOWLEDGE: dict[str, dict[str, Any]] = {
    "杭州": {
        "landmark": "西湖",
        "aliases": ["灵隐寺", "京杭大运河", "良渚古城遗址公园"],
        "culture": "宋韵文化、茶文化、数字经济城市气质",
        "color": "#1f6f68",
        "accent": "#d6ad60",
        "fact": "西湖文化景观于 2011 年列入世界文化遗产名录",
    },
    "宁波": {
        "landmark": "天一阁",
        "aliases": ["东钱湖", "象山影视城", "三江口"],
        "culture": "海丝文化、藏书文化、港口城市精神",
        "color": "#24526a",
        "accent": "#d8a45b",
        "fact": "天一阁是中国现存历史最悠久的私家藏书楼之一",
    },
    "温州": {
        "landmark": "雁荡山",
        "aliases": ["江心屿", "楠溪江", "泰顺廊桥"],
        "culture": "山水诗路、商贸文化、南戏传统",
        "color": "#315d4d",
        "accent": "#e5b866",
        "fact": "雁荡山以奇峰、飞瀑、洞壑和凝灰岩地貌闻名",
    },
    "绍兴": {
        "landmark": "鲁迅故里",
        "aliases": ["兰亭", "沈园", "东湖"],
        "culture": "越文化、书法文化、黄酒文化",
        "color": "#594a42",
        "accent": "#c98d54",
        "fact": "绍兴黄酒、兰亭书法和水乡街巷共同构成城市记忆",
    },
    "湖州": {
        "landmark": "南浔古镇",
        "aliases": ["莫干山", "太湖", "丝绸小镇"],
        "culture": "江南水乡、湖笔文化、生态文明",
        "color": "#2d6c63",
        "accent": "#e4bc70",
        "fact": "湖州是中国著名的湖笔产地，也是江南水乡的重要代表",
    },
    "嘉兴": {
        "landmark": "乌镇",
        "aliases": ["西塘古镇", "南湖", "月河历史街区"],
        "culture": "江南水乡、红船精神、古镇生活美学",
        "color": "#3c5f65",
        "accent": "#d5a861",
        "fact": "乌镇以河网、桥梁、白墙黛瓦和传统工艺构成独特水乡景观",
    },
    "金华": {
        "landmark": "婺州古城",
        "aliases": ["横店影视城", "双龙洞", "浦江水晶"],
        "culture": "婺剧、火腿饮食、影视文创",
        "color": "#654d3e",
        "accent": "#d4a25b",
        "fact": "婺剧是浙江重要地方戏曲，唱腔和脸谱具有鲜明地域特色",
    },
    "衢州": {
        "landmark": "江郎山",
        "aliases": ["衢州古城", "廿八都古镇", "孔氏南宗家庙"],
        "culture": "南孔文化、古道文化、山水生态",
        "color": "#35634e",
        "accent": "#e4b96d",
        "fact": "江郎山以三爿石奇峰和丹霞地貌景观著称",
    },
    "舟山": {
        "landmark": "普陀山",
        "aliases": ["东极岛", "嵊泗列岛", "朱家尖"],
        "culture": "海洋文化、观音文化、渔乡生活",
        "color": "#1c6072",
        "accent": "#f0c56c",
        "fact": "舟山群岛拥有海岛、渔港、寺院和海上日出的复合景观",
    },
    "台州": {
        "landmark": "神仙居",
        "aliases": ["府城墙", "天台山", "大陈岛"],
        "culture": "山海经纬、和合文化、海防记忆",
        "color": "#3e6550",
        "accent": "#e4b76b",
        "fact": "神仙居以火山流纹岩峰林、云海和峡谷景观见长",
    },
    "丽水": {
        "landmark": "古堰画乡",
        "aliases": ["云和梯田", "缙云仙都", "南尖岩"],
        "culture": "生态文明、畲族文化、乡村艺术",
        "color": "#477052",
        "accent": "#e6c274",
        "fact": "丽水以高森林覆盖率、梯田、古村和山水画乡构成生态名片",
    },
}


class OfflineLLMProvider(LLMProvider):
    name = "offline-zhejiang-editor"
    model = "zhejiang-editor-v1"

    # —— 候选池（offline：从知识库 imagery 生成 10 个确定性候选）——
    def candidates(self, context: dict[str, Any]) -> LLMResult:
        from ..services import shot_candidates as service

        city = kb.normalize_city(str(context.get("city") or ""))
        knowledge = CITY_KNOWLEDGE.get(city, {
            "landmark": city, "culture": "当地文化", "fact": f"走近{city}的山水与人文",
            "aliases": [], "color": "#2d6c63", "accent": "#d6ad60",
        })
        landmark = str(context.get("landmark") or knowledge["landmark"])
        entities = kb.topic_entities(city, landmark, str(context.get("theme") or ""),
                                     str(context.get("culture") or ""),
                                     str(context.get("festival") or "")) or []
        pool: list[tuple[str, list[str]]] = []
        for entity in entities[:6]:
            imagery = [str(item) for item in (entity.get("imagery") or []) if str(item).strip()]
            if imagery:
                pool.append((str(entity.get("name") or landmark), imagery))
        if not pool:
            pool = [(landmark, [f"{landmark}的山水轮廓", f"{landmark}的石材肌理", f"{landmark}的晨光投影"])]
        times = ["清晨", "正午", "黄昏", "夜"]
        sizes = ["远景", "全景", "中景", "近景", "特写"]
        target = service.target_shot_count(
            int(context.get("duration") or 30), int(context.get("shot_count") or 0)
        )
        total, role_counts = service.candidate_plan(target)
        roles = []
        for role in ("opening", "subject", "detail", "person", "closing"):
            roles.extend([role] * int(role_counts.get(role, 0)))
        candidates = []
        for index, role in enumerate(roles, start=1):
            name, imagery = pool[(index - 1) % len(pool)]
            time_value = times[(index - 1) % len(times)]
            size_value = sizes[(index - 1) % len(sizes)]
            place = f"{city}{name}" if city and city not in name else name
            candidates.append(
                {
                    "id": f"c{index}",
                    "role": role,
                    "title": f"{name}·{size_value}{index}",
                    "hook": f"{time_value}的{name}，{imagery[0]}",
                    "imagery": imagery[:3],
                    "place": place,
                    "time_of_day": time_value,
                    "shot_size": size_value,
                    "motion": "环境细节缓慢变化",
                    "why": f"以{size_value}呈现{name}，建立{city}的辨识度",
                    "duration_hint": 5,
                }
            )
        return LLMResult(
            data={
                "arc": {
                    "opening": f"抵达{city}，建立空间与时间",
                    "development": f"深入{landmark}的细节与人文",
                    "climax": f"{landmark}最具辨识度的一刻",
                    "closing": "留白收束",
                },
                "candidates": candidates,
            },
            provider=self.name,
            model=self.model,
            fallback_used=True,
            event={
                "type": "shot_candidates",
                "provider": self.name,
                "model": self.model,
                "count": len(candidates),
                "target_shots": target,
                "rewrite_attempts": 0,
                "prompt_version": "candidates-offline-v1",
            },
        )

    # —— 逐镜详述（offline：把候选字段补齐成可过校验门的规格）——
    def detail(self, context: dict[str, Any]) -> dict[str, Any]:
        selected = list(context.get("selected") or [])
        per_shot_seconds = [float(value) for value in (context.get("per_shot_seconds") or [])]
        task_input = dict(context.get("task_input") or {})
        duration = int(task_input.get("duration", 30))
        budget = plan_narration_budget(duration, len(selected))
        per_chars = int(budget["per_shot_chars"])
        shots = []
        for index, (row, seconds) in enumerate(zip(selected, per_shot_seconds), start=1):
            place = str(row.get("place") or task_input.get("landmark") or task_input.get("city") or "")
            time_value = str(row.get("time_of_day") or "清晨")
            if time_value not in ("清晨", "正午", "黄昏", "夜"):
                time_value = "清晨"
            light = {
                "清晨": ("右上方", "低角度散射光", "低"),
                "正午": ("正上方", "顶光硬光", "高"),
                "黄昏": ("左后方", "暖色侧光", "中"),
                "夜": ("侧前方", "低照度暖光点", "高"),
            }[time_value]
            size_value = str(row.get("shot_size") or "中景")
            if size_value not in ("远景", "全景", "中景", "近景", "特写"):
                size_value = "中景"
            anchors = [str(item) for item in (row.get("imagery") or []) if str(item)]
            first = anchors[0] if anchors else "石材肌理"
            last = anchors[-1] if anchors else "水痕"
            detail_extra = f"{first}与{last}的细节层次"
            must_include = (anchors + [detail_extra])[:5]
            anchor_text = "、".join(anchors[:3]) if anchors else first
            body = (
                f"{place}的{row.get('title') or '主景'}，{anchor_text}清楚可见；"
                f"{time_value}{light[1]}，光线从{light[0]}来，明暗过渡平缓；"
                f"{size_value}平视，主体位于画面中部偏下，前景留出环境，上方留白给字幕；"
                "冷灰色调，青灰与雾白为主，暗部保留细节；"
                f"{row.get('motion') or '环境细节缓慢流动'}，水面与石面的反光轻轻晃动，雾气缓慢流动。"
            )
            # offline 旁白：从锚点与地点拼句并补到字数预算（模板腔，仅用于离线跑通）
            base = f"{place}，{first}在{time_value}里显出层次。"
            filler = f"{last}停在画面里，脚步放慢，看得更清。"
            narration = (base + filler)[:per_chars]
            while len(narration) < max(9, per_chars - 6):
                narration += "再看一眼。"
            narration = narration[:per_chars]
            shots.append(
                {
                    "candidate_id": row.get("candidate_id") or row.get("id"),
                    "duration_sec": round(seconds, 2),
                    "subject": {
                        "name": str(row.get("title") or place),
                        "material": "青灰石材与水汽",
                        "details": [first, last],
                        "scale": size_value,
                    },
                    "environment": {
                        "place": place,
                        "season": "春",
                        "weather": "薄雾" if time_value == "清晨" else "晴",
                        "time": time_value,
                    },
                    "lighting": {"direction": light[0], "quality": light[1], "contrast": light[2]},
                    "composition": {
                        "shot_size": size_value,
                        "angle": "平视",
                        "framing": "主体居中，前景留出环境",
                        "negative_space": "上方留白给字幕",
                    },
                    "palette": {"temperature": "冷", "dominant": ["青灰", "雾白"]},
                    "motion": {"in_frame": [row.get("motion") or "环境细节缓慢变化"], "camera": "静止"},
                    "must_include": must_include,
                    "must_avoid": ["文字", "现代广告牌"],
                    "continuity": {
                        "same_subject": True,
                        "same_palette": True,
                        "relation_to_prev": "延续上一镜的色调，视角略作变化" if index > 1 else "",
                    },
                    "visual_prompt": body,
                    "narration": narration,
                    "subtitle": narration,
                }
            )
        from ..services import shot_detail as detail_service

        normalized = detail_service.normalize_detail(
            {"shots": shots}, selected=selected, per_shot_seconds=per_shot_seconds
        )
        gate = detail_service.validate_detail(
            normalized,
            selected=selected,
            theme=str(task_input.get("theme") or ""),
            city=str(task_input.get("city") or ""),
            narration_budget=budget,
            composed=detail_service.build_composed_prompts(
                normalized, dict(context.get("style_ctx") or {})
            ),
        )
        return {
            "title": f"{task_input.get('city') or ''}{task_input.get('theme') or ''}".strip() or "候选成片",
            "summary": f"{task_input.get('theme') or '主题'}：按所选镜头顺序成片。",
            "cta": "来走走看看。",
            "shots": normalized,
            "gate": {
                "ok": gate.ok,
                "problems": gate.problems,
                "composed_lengths": gate.composed_lengths,
                "rewrite_attempts": 0,
            },
            "event": {
                "type": "shot_detail",
                "provider": self.name,
                "model": self.model,
                "shot_count": len(normalized),
                "gate_ok": gate.ok,
                "problem_count": len(gate.problems),
                "rewrite_attempts": 0,
                "prompt_version": "detail-offline-v1",
            },
        }

    def generate(self, prompt: str, context: dict[str, Any]) -> LLMResult:
        city = kb.normalize_city(context["city"])
        knowledge = CITY_KNOWLEDGE.get(city, {
            "landmark": city, "culture": "当地文化", "fact": f"走近{city}的山水与人文",
            "aliases": [], "color": "#2d6c63", "accent": "#d6ad60",
        })
        landmark = context.get("landmark") or knowledge["landmark"]
        theme = context["theme"]
        culture = context.get("culture") or knowledge["culture"]
        festival = context.get("festival") or "当季文旅推广"
        audience = context.get("audience") or "年轻游客"
        duration = int(context.get("duration", 30))
        style = context.get("style", "诗意纪实")
        shot_count = int(context.get("shot_count") or (4 if duration <= 15 else 6 if duration <= 30 else 8))
        total_unit = duration / shot_count
        entries = kb.topic_entities(city, landmark, theme, culture, context.get("festival", ""))
        evidence = []
        sentences = []
        for i in range(kb.FACTS_PER_ENTITY):
            for entity in entries:
                facts = entity["facts"]
                if i < len(facts) and len(evidence) < shot_count - 1:
                    evidence.append(entity)
                    sentences.append(facts[i].rstrip("。") + "。")
        hooks = [
            f"如果把一座城市写成一首诗，{city}的第一句，往往从{landmark}开始。",
            f"沿着水脉与山色走进{city}，你会遇见{culture}留下的时间纹理。",
            f"这里不只是一处风景，更是一场属于{audience}的城市发现。",
        ]
        title = f"{city}，把{theme}写进山水之间"
        summary = f"围绕{theme}，以{landmark}为视觉入口，讲述{city}的{culture}与当季{festival}。"
        narration = (
            f"{hooks[0]} "
            f"清晨的光落在{landmark}，古老的街巷、桥影和远山慢慢苏醒。 "
            f"{knowledge['fact']}。 "
            f"从一杯茶、一味风物，到一段仍在延续的手艺，{city}把过去和今天放在同一条旅行路上。 "
            f"这个{festival}，来{city}走一走，让风景成为记忆，让每一次出发都值得被看见。"
        )
        visual_templates = [
            ("城市开场", f"{city}城市意象与{landmark}远景，晨雾、柔和日光、竖屏构图、{style}"),
            ("地标入画", f"{landmark}的代表性景观，水面或山体前景，游客剪影，镜头缓慢推进"),
            ("文化细节", f"{culture}相关的手作、街巷、器物或生活细节，特写，真实质感"),
            ("风物体验", f"{city}地方风物与市井烟火，摊铺、茶点、手艺，温暖自然光"),
            ("当季活动", f"{festival}氛围中的城市公共空间，灯笼、花事或人群，节奏明快"),
            ("收束号召", f"{city}山水与城市夜色交叠，留出标题空间，适合文旅宣传片结尾"),
            ("隐藏路线", f"{landmark}周边小众步道或古村，轻松旅行视角，空气感"),
            ("品牌收尾", f"{city}城市天际线与自然景观融合，电影感，适合品牌口号"),
        ]
        shots: list[dict[str, Any]] = []
        for index in range(shot_count):
            label, visual = visual_templates[index]
            start = round(index * len(narration) / shot_count)
            end = round((index + 1) * len(narration) / shot_count)
            text = narration[start:end].strip("，。 ") + "。"
            if evidence:
                if index < len(evidence):
                    entity = evidence[index]
                    label = entity["name"]
                    imagery = entity.get("imagery") or [label]
                    visual = f"{city}·{label}，{imagery[index % len(imagery)]}，竖屏构图"
                    text = sentences[index]
                else:
                    entry = entries[0]
                    label = entry["name"]
                    imagery = entry.get("imagery") or [label]
                    visual = f"{city}·{label}，{imagery[index % len(imagery)]}，竖屏构图"
                    text = f"沿着{label}，读懂{city}。" if index == shot_count - 1 else f"下一站，走进{city}的{label}。"
            shots.append(
                {
                    "id": f"shot_{index + 1}",
                    "order": index + 1,
                    "duration_sec": round(total_unit, 2),
                    "title": label,
                    "visual_prompt": visual,
                    "narration": text,
                    "subtitle": text,
                    "transition": "fade" if index == 0 else ("dissolve" if index % 2 else "slideleft"),
                }
            )
        tags = [city, landmark, culture, festival, audience, style]
        data = {
            "title": title,
            "summary": summary,
            "cta": f"关注浙里成片，下一站，去{city}。",
            "tags": [tag for tag in tags if tag],
            "narration": "".join(shot["narration"] for shot in shots),
            "shots": shots,
            "knowledge": {
                "fact": sentences[0] if sentences else knowledge["fact"],
                "color": knowledge["color"],
                "accent": knowledge["accent"],
                "aliases": knowledge["aliases"],
            },
        }
        event = {
            "type": "llm",
            "provider": self.name,
            "model": self.model,
            "prompt_version": "zj-tourism-prompt-v2",
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "ok": True,
            "cost_basis": "no_external_charge",
        }
        return LLMResult(data=data, provider=self.name, model=self.model, fallback_used=True, event=event)
