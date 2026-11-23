/* 分步工作台：五步向导（需求 → 选镜头 → 审核脚本 → 审核 prompt → 生成验收）。
 *
 * 设计要点：
 * - 每一步只渲染当前内容，切步时回到页面顶部，避免单页越滚越长；
 * - 已完成步骤可点回退，下游内容保留并标"已过期"，不自动删除；
 * - 步骤可用性由 task.status / task.shots 推导，不做"跳到一半没数据"的允许。
 *
 * 本文件先于 app.js 加载，函数体在调用时才访问 app.js 的全局（state/escapeHtml 等）。
 */

const WIZARD_STEPS = [
  { id: 1, label: "需求", hint: "城市、主题、时长与创作参数" },
  { id: 2, label: "选镜头", hint: "从候选方向里选择并排序" },
  { id: 3, label: "审核脚本", hint: "旁白、字幕、配音与音乐" },
  { id: 4, label: "审核 prompt", hint: "逐镜画面规格与生成 prompt" },
  { id: 5, label: "生成验收", hint: "生成视频并做质量验收" },
];

function wizardStepFor(task) {
  if (!task || !task.id) return 1;
  if (["queued", "running", "completed"].includes(task.status)) return 5;
  if (task.status === "candidates_ready" || task.status === "selected") return 2;
  if (!Array.isArray(task.shots) || !task.shots.length) return 2;
  // 2–5 都允许：草稿态也能回退到"选镜头"重选（下游保留并标记过期）。
  // 未显式指定时默认落在 3（审核脚本）。
  const desired = Number(state.wizardStep) || 3;
  return Math.min(5, Math.max(1, desired));
}

function wizardStepAvailable(task, step) {
  if (step === 1) return true;
  if (!task || !task.id) return false;
  if (step === 2) return !["queued", "running", "completed"].includes(task.status);
  if (step === 3 || step === 4 || step === 5) {
    return Array.isArray(task.shots) && task.shots.length > 0;
  }
  return false;
}

function wizardStepperMarkup(task, current) {
  return `<nav class="wizard-stepper" aria-label="创作步骤">${WIZARD_STEPS.map((step) => {
    const available = wizardStepAvailable(task, step.id);
    const done = step.id < current;
    const classes = ["wizard-step", step.id === current ? "current" : "", done ? "done" : ""].filter(Boolean).join(" ");
    return `<button type="button" class="${classes}" data-wizard-goto="${step.id}" ${available ? "" : "disabled"} ${step.id === current ? 'aria-current="step"' : ""}>
      <b>${String(step.id).padStart(2, "0")}</b>
      <span>${escapeHtml(step.label)}</span>
      <small>${escapeHtml(step.hint)}</small>
    </button>`;
  }).join("")}</nav>`;
}

function wizardNavMarkup(task, current) {
  const prev = current > 1
    ? `<button type="button" class="text-button wizard-prev" data-wizard-goto="${current - 1}">← 上一步</button>`
    : `<span class="wizard-nav-hint">从左侧表单开始，或用作品档案打开已有任务。</span>`;
  let next = "";
  if (current === 2) {
    next = `<span class="wizard-nav-hint">选好后点右侧「确认选择并生成详述」进入下一步。</span>`;
  } else if (current === 3) {
    next = `<button type="button" class="primary-button" data-wizard-next="4">下一步：审核 prompt<b>→</b></button>`;
  } else if (current === 4) {
    next = `<button type="button" class="primary-button" data-wizard-next="5">下一步：生成验收<b>→</b></button>`;
  } else if (current === 5) {
    const running = ["queued", "running"].includes(task.status);
    const canGenerate = ["draft", "edited", "failed"].includes(task.status) && (task.shots || []).length > 0;
    next = running
      ? `<span class="wizard-nav-hint">正在生成，进度见上方。</span>`
      : canGenerate
        ? `<button type="button" class="primary-button" data-generate-task="${escapeHtml(task.id)}">确认并生成实际视频<b>→</b></button>`
        : `<a class="primary-button wizard-download" href="#view-projects" data-open-archive>查看作品档案</a>`;
  }
  return `<div class="wizard-nav">${prev}<div class="wizard-nav-next">${next}</div></div>`;
}

function staleBannerMarkup(task) {
  // 用户回退改了镜头选择、但还没重新生成详述时，下游内容全部来自上一版选择。
  // 按事件顺序判断：最后一次选择晚于最后一次详述 → 下游过期。
  const events = task.trace?.events || [];
  let lastSelection = -1;
  let lastDetail = -1;
  events.forEach((event, index) => {
    if (event.type === "shot_selection" && event.stale_downstream) lastSelection = index;
    if (event.type === "shot_detail") lastDetail = index;
  });
  if (lastSelection < 0 || lastSelection < lastDetail) return "";
  return `<p class="stale-banner" role="status">镜头选择已变更：当前脚本 / prompt / 素材来自上一版选择。请点「重新生成全部 prompt」，或在生成页重新生成视频。</p>`;
}

function stepBodyMarkup(task, step) {
  if (step === 1) return stepBriefMarkup(task);
  if (step === 2) return candidateSelectionMarkup(task);
  if (step === 3) return staleBannerMarkup(task) + stepScriptMarkup(task);
  if (step === 4) return staleBannerMarkup(task) + stepPromptMarkup(task);
  return staleBannerMarkup(task) + stepRenderMarkup(task);
}

function stepBriefMarkup(task) {
  const input = task.input || {};
  const rows = [
    ["城市", input.city], ["核心地标", input.landmark || "未指定"], ["主题", input.theme],
    ["叙事风格", input.style], ["时长", `${input.duration} 秒`],
    ["镜头数", Number(input.shot_count) ? `${input.shot_count} 个` : "自动"],
    ["素材来源", input.asset_source], ["配音", input.voice],
    ["背景音乐", input.music_track_id || input.music_mood],
  ];
  return `<section class="wizard-panel">
    <div class="wizard-head"><span class="section-index">01 / BRIEF</span><h4>需求</h4>
      <p>这组参数已经用于生成候选镜头。改需求会新建一个任务，旧任务保留在作品档案里。</p></div>
    <div class="brief-grid">${rows.map(([label, value]) => `<div><span>${escapeHtml(label)}</span><strong>${escapeHtml(String(value ?? "—"))}</strong></div>`).join("")}</div>
    <div class="wizard-actions"><button type="button" class="text-button" data-brief-restart>修改需求并新建任务</button></div>
  </section>`;
}

function scriptSoundMarkup(task) {
  const input = task.input;
  const draft = (key) => `class="draft-input" data-input-field="${key}"`;
  return `<section class="review-block">
    <div class="review-block-head"><span class="section-index">03 / VOICE</span><h5>念什么，配什么</h5></div>
    <div class="review-settings-grid">
      <label>配音<select ${draft("voice")}>${voiceChoiceOptions(input.voice)}</select></label>
      <label>语速<input ${draft("voice_rate")} type="number" min="0.5" max="2" step="0.1" value="${input.voice_rate}" /></label>
      <label>音高 (Hz)<input ${draft("voice_pitch")} type="number" min="-50" max="50" step="5" value="${input.voice_pitch ?? 0}" /></label>
      <label>配音音量<input ${draft("voice_volume")} type="number" min="0" max="2" step="0.05" value="${input.voice_volume}" /></label>
      <label>背景音乐<select ${draft("music_mood")}>${choiceOptions(state.catalog?.music_moods || [], input.music_mood)}</select></label>
      <label>音乐音量<input ${draft("music_volume")} type="number" min="0" max="1" step="0.01" value="${input.music_volume}" /></label>
      <label>旁白出现时<select ${draft("music_ducking")}>${(state.catalog?.music_ducking || ["不压低", "轻微", "标准"]).map((value) => `<option value="${escapeHtml(value)}" ${input.music_ducking === value ? "selected" : ""}>${escapeHtml(value === "不压低" ? "保持音量不变" : value + "压低")}</option>`).join("")}</select></label>
      <label class="toggle-label">烧录字幕<input ${draft("include_subtitles")} type="checkbox" ${input.include_subtitles ? "checked" : ""} /><span class="toggle"></span></label>
      <label class="toggle-label">AIGC 标识<input ${draft("include_ai_label")} type="checkbox" ${input.include_ai_label ? "checked" : ""} /><span class="toggle"></span></label>
    </div>
    ${voiceStudioMarkup(task)}
    ${voiceAttachPanel(task)}
    ${musicAttachPanel(task)}
  </section>`;
}

function promptSettingsMarkup(task) {
  const input = task.input;
  const draft = (key) => `class="draft-input" data-input-field="${key}"`;
  return `<section class="review-block">
    <div class="review-block-head"><span class="section-index">04 / PICTURE</span><h5>画面从哪来</h5></div>
    <label class="source-picker">素材来源<select ${draft("asset_source")}${visibleAssetSource(input.asset_source) !== input.asset_source ? ` data-saved-source="${escapeHtml(input.asset_source)}"` : ""}>${assetProviderOptions(input.asset_source)}</select></label>
    <label class="source-picker">使用模式<select ${draft("visual_usage")}><option value="preview" ${input.visual_usage === "preview" ? "selected" : ""}>技术预览（允许待审素材）</option><option value="publication" ${input.visual_usage === "publication" ? "selected" : ""}>正式发布（仅审核通过）</option></select></label>
    <p class="section-note" id="source-note">${escapeHtml(describeSource(input.asset_source))}</p>
    ${assetAttachPanel(task)}
    <div class="review-settings-grid">
      <label>输出规格<select ${draft("output_resolution")}>${["720x1280", "1080x1920", "1920x1080", "1080x1080"].map((value) => `<option value="${value}" ${input.output_resolution === value ? "selected" : ""}>${value}</option>`).join("")}</select></label>
      <label>帧率<select ${draft("fps")}>${[24, 25, 30, 60].map((value) => `<option value="${value}" ${Number(input.fps) === value ? "selected" : ""}>${value} fps</option>`).join("")}</select></label>
    </div>
    <label>主体锁定<textarea ${draft("subject_lock")} placeholder="同一座断桥、同一件器物或同一位人物的固定外观">${escapeHtml(input.subject_lock || "")}</textarea><small class="field-note">逐镜复用主体描述；填写参考图路径时可进一步启用图像条件。</small></label>
    <label>主体参考图路径<input ${draft("subject_reference_path")} value="${escapeHtml(input.subject_reference_path || "")}" placeholder="可选：本地图片绝对路径" /></label>
    <label>补充要求<textarea ${draft("custom_brief")}>${escapeHtml(input.custom_brief)}</textarea></label>
  </section>`;
}

function stepScriptMarkup(task) {
  return `<section class="wizard-panel">
    <div class="wizard-head"><span class="section-index">03 / SCRIPT</span><h4>审核脚本与声音</h4>
      <p>改完点「保存修改」。旁白决定配音时长，字幕必须与旁白逐字对应。</p></div>
    <label>标题<input class="draft-title" value="${escapeHtml(task.title)}" /></label>
    <label>摘要<textarea class="draft-summary">${escapeHtml(task.summary)}</textarea></label>
    <label>CTA<textarea class="draft-cta">${escapeHtml(task.cta)}</textarea></label>
    <label>完整旁白<textarea class="draft-narration">${escapeHtml(task.narration)}</textarea></label>
    ${scriptSoundMarkup(task)}
    <section class="review-block">
      <div class="review-block-head"><span class="section-index">03 / SHOTS</span><h5>逐镜要念的、要拍的</h5></div>
      <div class="shot-editor-grid">${shotScriptEditors(task)}</div>
    </section>
    <button class="text-button" data-save-draft="${escapeHtml(task.id)}">保存脚本与声音修改</button>
  </section>`;
}

function stepPromptMarkup(task) {
  return `<section class="wizard-panel">
    <div class="wizard-head"><span class="section-index">04 / PROMPT</span><h4>审核画面 prompt</h4>
      <p>每镜的完整 prompt 就是实际发给生成模型的文字。锚点会被语义验收逐条核对覆盖率。</p></div>
    ${detailGateMarkup(task)}
    ${promptSettingsMarkup(task)}
    <section class="review-block">
      <div class="review-block-head"><span class="section-index">04 / SHOTS</span><h5>逐镜画面规格</h5></div>
      <div class="shot-editor-grid">${shotPromptEditors(task)}</div>
    </section>
    ${libraryPanelMarkup(task)}
    <div class="wizard-actions">
      <button class="text-button" data-regenerate-detail="${escapeHtml(task.id)}">重新生成全部 prompt</button>
      <button class="text-button" data-save-draft="${escapeHtml(task.id)}">保存修改</button>
    </div>
  </section>`;
}

function stepRenderMarkup(task) {
  const render = task.render || {};
  const error = task.error ? `<div class="error-box"><strong>任务失败</strong><p>${escapeHtml(task.error)}</p><button class="text-button" data-retry="${escapeHtml(task.id)}">从脚本阶段重试</button></div>` : "";
  const mediaLinks = render.video_url ? `<a href="${render.video_url}" target="_blank">打开 MP4</a><a href="${render.video_url}" download>下载 MP4</a><a href="${render.subtitle_url}" target="_blank">下载 SRT</a>${task.audio?.voice_url ? `<a href="${task.audio.voice_url}" target="_blank">试听配音</a>` : ""}${task.audio?.music_url ? `<a href="${task.audio.music_url}" target="_blank">试听音乐</a>` : ""}` : "";
  return `<section class="wizard-panel">
    <div class="wizard-head"><span class="section-index">05 / RENDER</span><h4>生成与验收</h4>
      <p>${["queued", "running"].includes(task.status) ? "正在生成，下面的进度会实时更新。" : "生成完成后，这里会出现播放器与两层验收结果。"}</p></div>
    ${error}
    <div class="task-links"><a href="/api/tasks/${encodeURIComponent(task.id)}/export?format=md" download>导出文案</a>${mediaLinks}</div>
    ${render.video_url ? `<video class="result-video" controls playsinline src="${render.video_url}"></video>` : ""}
    ${assetList(task)}
    ${semanticQaMarkup(task)}
    ${visualQaMarkup(task)}
    ${visualReviewMarkup(task)}
    ${billingMarkup(task)}
    ${(task.audio?.subject_reviews || []).map((sheet) => sheet.url ? `<a href="${escapeHtml(sheet.url)}" target="_blank">复核 ${escapeHtml(sheet.group)} · 首中尾帧</a>` : escapeHtml(sheet.error)).join(" ")}
    ${task.audio?.subtitle_alignment ? '<p>字幕对齐 · ' + task.audio.subtitle_alignment.segments.map((segment) => escapeHtml(segment.shot_id + " " + (segment.source === "sentence" ? "句级（" + segment.reason + "）" : segment.source))).join(" / ") + "</p>" : ""}
    <p>${escapeHtml(task.audio?.music_notice || (task.audio?.music_match?.id ? (task.audio.music_match.title || task.audio.music_match.id) + " · " + task.audio.music_match.license : ""))}</p>
    <div class="task-links"><a href="/api/tasks/${encodeURIComponent(task.id)}/ledger" download>下载任务底账</a></div>
    <div class="trace-line">文本 provider: ${escapeHtml(task.trace?.provider || "pending")} / 模型: ${escapeHtml(task.trace?.model || "pending")} / fallback: ${String(Boolean(task.trace?.fallback_used))}<br />配音: ${escapeHtml(task.audio?.voice_provider || "pending")} / 音乐: ${escapeHtml(task.audio?.music_provider || "pending")} / 音频: ${render.has_audio ? "有" : "无"} / 烧录字幕: ${render.has_burned_subtitles ? "有" : "无"}<br />Provider 报告成本: ${Number(task.cost?.total || 0).toFixed(4)}（${escapeHtml(task.cost?.unit || "等待统计")}）</div>
  </section>`;
}

function shotScriptEditors(task) {
  return (task.shots || []).map((shot) => `<article class="shot-editor">
    <div class="shot-editor-header"><strong>镜头 ${String(shot.order).padStart(2, "0")}</strong><span>${escapeHtml(shot.id)}</span></div>
    <div class="shot-editor-row">
      <label>知识实体<select data-shot-field="entity_id" data-shot-id="${escapeHtml(shot.id)}">${entityOptions(task.input.city, shot.entity_id)}</select></label>
      <label>镜头意图<select data-shot-field="shot_tag" data-shot-id="${escapeHtml(shot.id)}">${shotTagOptions(shot.shot_tag)}</select></label>
    </div>
    <label>镜头标题<input data-shot-field="title" data-shot-id="${escapeHtml(shot.id)}" value="${escapeHtml(shot.title)}" /></label>
    <label>旁白<textarea data-shot-field="narration" data-shot-id="${escapeHtml(shot.id)}">${escapeHtml(shot.narration)}</textarea></label>
    <label>字幕<textarea data-shot-field="subtitle" data-shot-id="${escapeHtml(shot.id)}">${escapeHtml(shot.subtitle)}</textarea></label>
    <div class="shot-editor-row">
      <label>进入转场<select data-shot-field="transition" data-shot-id="${escapeHtml(shot.id)}">${[["fade","淡变"],["dissolve","叠化"],["smoothleft","平滑左移"],["smoothright","平滑右移"],["circleopen","圆形展开"],["wipeleft","向左划像"],["wiperight","向右划像"],["zoom","推近"],["slideleft","左移（兼容）"],["slideright","右移（兼容）"]].map(([value, label]) => `<option value="${value}" ${shot.transition === value ? "selected" : ""}>${label}</option>`).join("")}</select></label>
      <label>时长（秒）<input type="number" min="1" max="20" step="0.1" data-shot-field="duration_sec" data-shot-id="${escapeHtml(shot.id)}" value="${shot.duration_sec}" /></label>
    </div>
  </article>`).join("");
}

function specSummaryMarkup(shot) {
  const spec = shot.prompt_spec || {};
  const subject = spec.subject || {};
  const environment = spec.environment || {};
  const lighting = spec.lighting || {};
  const composition = spec.composition || {};
  const palette = spec.palette || {};
  const motion = spec.motion || {};
  const rows = [
    ["主体", [subject.name, subject.material, (subject.details || []).join("、")].filter(Boolean).join(" / ")],
    ["环境光线", [environment.time, environment.place, lighting.direction, lighting.quality].filter(Boolean).join(" / ")],
    ["构图", [composition.shot_size, composition.angle, composition.framing, composition.negative_space].filter(Boolean).join(" / ")],
    ["色彩", [palette.temperature, (palette.dominant || []).join("、")].filter(Boolean).join(" / ")],
    ["动态", [(motion.in_frame || []).join("、"), motion.camera].filter(Boolean).join(" / ")],
  ].filter(([, value]) => value);
  if (!rows.length) return "";
  return `<details class="spec-summary"><summary>结构化规格（只读）</summary><div>${rows.map(([label, value]) => `<p><span>${escapeHtml(label)}</span>${escapeHtml(value)}</p>`).join("")}</div></details>`;
}

function shotPromptEditors(task) {
  return (task.shots || []).map((shot) => `<article class="shot-editor">
    <div class="shot-editor-header"><strong>镜头 ${String(shot.order).padStart(2, "0")}</strong><span>${escapeHtml(shot.id)}</span></div>
    <label>画面描述<textarea data-shot-field="visual_prompt" data-shot-id="${escapeHtml(shot.id)}">${escapeHtml(shot.visual_prompt)}</textarea></label>
    <label>意图锚点（顿号分隔）<input data-shot-field="intent_anchors" data-shot-id="${escapeHtml(shot.id)}" value="${escapeHtml((shot.intent_anchors || []).join("、"))}" /></label>
    <label>本镜回避（顿号分隔）<input data-shot-field="must_avoid" data-shot-id="${escapeHtml(shot.id)}" value="${escapeHtml((shot.must_avoid || []).join("、"))}" /></label>
    ${(shot.intent_anchors || []).length ? `<div class="shot-anchors"><span>意图锚点</span>${shot.intent_anchors.map((anchor) => `<em>${escapeHtml(anchor)}</em>`).join("")}</div>` : ""}
    ${composedPromptMarkup(shot)}
    ${specSummaryMarkup(shot)}
  </article>`).join("");
}

function bindWizard(task) {
  document.querySelectorAll("[data-wizard-goto]").forEach((button) => {
    button.addEventListener("click", () => goToWizardStep(task, Number(button.dataset.wizardGoto)));
  });
  document.querySelectorAll("[data-wizard-next]").forEach((button) => {
    button.addEventListener("click", async () => {
      const next = Number(button.dataset.wizardNext);
      const saved = await saveDraft(task.id, { refresh: false, notify: false });
      if (saved === false) return;
      goToWizardStep(task, next, { afterSave: true });
      showToast(next === 4 ? "已保存，进入 prompt 审核。" : "已保存，进入生成与验收。");
    });
  });
  document.querySelectorAll("[data-brief-restart]").forEach((button) => {
    button.addEventListener("click", () => restartFromBrief(task));
  });
  const regenerate = document.querySelector("[data-regenerate-detail]");
  if (regenerate) regenerate.addEventListener("click", () => regenerateDetail(task));
  document.querySelectorAll("[data-open-archive]").forEach((link) => {
    link.addEventListener("click", (event) => {
      event.preventDefault();
      showView("projects");
    });
  });
}

async function goToWizardStep(task, step, options = {}) {
  if (!wizardStepAvailable(task, step)) return;
  state.wizardStep = step;
  if (options.afterSave) {
    const refreshed = await api(`/api/tasks/${encodeURIComponent(task.id)}`);
    renderTask(refreshed);
  } else {
    renderTask(task);
  }
  window.scrollTo({ top: 0, behavior: "smooth" });
}

function restartFromBrief(task) {
  // 回到表单等于开新任务：清掉地址栏里的旧任务深链，避免刷新又跳回旧任务
  if (window.location.hash) history.replaceState(null, "", window.location.pathname + window.location.search);
  showView("create");
  prefillCreateForm(task.input || {});
  state.activeTask = null;
  state.wizardStep = null;
  $("#active-task").innerHTML = `<p class="empty-state">修改需求后点「生成镜头方向」，会创建一个新任务；旧任务保留在作品档案。</p>`;
  setCreateFormVisible(true);
  showToast("已把当前需求填回表单，提交后会新建任务。");
}

function setCreateFormVisible(visible) {
  const workspace = $("#new-task-workspace");
  if (workspace) workspace.hidden = !visible;
}

function prefillCreateForm(input) {
  const form = $("#create-form");
  if (!form) return;
  const fields = ["city", "landmark", "theme", "culture", "festival", "audience", "duration", "shot_count",
    "style", "asset_source", "output_resolution", "fps", "voice", "music_mood", "voice_rate",
    "voice_pitch", "voice_volume", "music_volume", "music_ducking", "subject_lock", "custom_brief"];
  for (const key of fields) {
    const field = form.elements[key];
    if (!field || input[key] === undefined || input[key] === null) continue;
    const value = String(input[key]);
    if (field.tagName === "SELECT") {
      const option = Array.from(field.options).find((item) => item.value === value);
      if (option) field.value = value;
    } else {
      field.value = value;
    }
  }
  if (form.elements.include_subtitles) form.elements.include_subtitles.checked = Boolean(input.include_subtitles);
  if (form.elements.include_ai_label) form.elements.include_ai_label.checked = Boolean(input.include_ai_label);
}

async function regenerateDetail(task) {
  try {
    showToast("正在按当前选择重新生成 prompt…");
    const result = await api(`/api/tasks/${encodeURIComponent(task.id)}/detail`, { method: "POST", body: "{}" });
    const refreshed = await api(`/api/tasks/${encodeURIComponent(task.id)}`);
    renderTask(refreshed);
    const problems = result?.gate?.problems || [];
    showToast(problems.length ? `已重新生成，仍有 ${problems.length} 项校验提示` : "已重新生成，校验通过");
  } catch (error) {
    showToast(error.message);
  }
}
