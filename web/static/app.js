const baseList = document.getElementById('base-variable-list');
const derivedList = document.getElementById('derived-variable-list');
const probabilityWrap = document.getElementById('probability-table-wrap');
const sumBanner = document.getElementById('probability-sum-banner');
const statusBanner = document.getElementById('status-banner');
const summaryCards = document.getElementById('summary-cards');
const variableSummary = document.getElementById('variable-summary');
const distributionPreview = document.getElementById('distribution-preview');
const matrixRoot = document.getElementById('matrix-root');
const resultsRoot = document.getElementById('results-root');
const explanationRoot = document.getElementById('explanation-root');
const codeRoot = document.getElementById('code-root');
const formulaInput = document.getElementById('formula-input');
const formulaChipBank = document.getElementById('formula-chip-bank');
const activeFormulaBank = document.getElementById('active-formula-bank');
const normalizeBeforeSubmit = document.getElementById('normalize-before-submit');
const langSwitch = document.getElementById('lang-switch');

let lastResult = null;
let currentLang = 'zh';
const selectedFormulas = new Set();
const matrixState = { rowAxis: '', colAxis: '' };

const I18N = {
    zh: {
        hero_title: 'pyShannon 信息论计算器',
        hero_text: '用现有 Python 计算核心驱动网页交互。支持随机变量、派生变量、联合分布输入，自动计算熵、条件熵、互信息和条件互信息，还能递归展开公式依赖与对应代码。',
        workspace_title: '变量工作台',
        workspace_text: '先定义基础随机变量，再按需要添加派生变量。',
        base_vars_title: '基础变量',
        derived_vars_title: '派生变量',
        joint_table_title: '联合概率表',
        result_title: '计算结果',
        result_text: '结果完全由后端 Python 计算产生，前端负责筛选、递归展开和代码展示。',
        add_variable: '新增变量',
        add_derived_variable: '新增派生变量',
        build_table: '生成概率表',
        load_sample: '加载题图示例',
        fill_uniform: '均匀填充',
        normalize_table: '当前表归一化',
        normalize_before_submit: '计算前自动归一化',
        run_calculation: '开始计算',
        exam_hint: '提示：如果你只想复现题图，直接点“加载题图示例”再点“开始计算”就可以。',
        formula_placeholder: '输入公式并点击添加，例如 H(X|Y), I(X;Y|Z)',
        add_formula_focus: '添加聚焦公式',
        clear_formula_focus: '清空聚焦',
        base_pill: '基础',
        derived_pill: '派生',
        remove: '移除',
        field_key: '键名',
        field_latex: 'LaTeX',
        field_states: '状态',
        field_expression: '表达式',
        preview: '预览',
        placeholder_key: 'X',
        placeholder_latex: 'X',
        placeholder_states: '0,1',
        placeholder_expression: 'X*Y',
        waiting_input: '等待输入',
        need_base_var: '先添加至少一个基础变量。',
        invalid_states: (key) => `变量 ${key || '?'} 需要至少两个状态。`,
        probability_sum: (sum) => `当前概率和：${sum.toFixed(6)}`,
        table_rebuilt: '概率表已根据当前基础变量重建。',
        normalize_zero_error: '当前概率表总和为 0，暂时不能归一化。',
        normalized_ok: '当前概率表已归一化。',
        sample_loaded: '题图示例已载入，可以直接点击“开始计算”。',
        formula_not_found: (text) => `没有找到公式 ${text}。请直接点击公式芯片，或输入精确公式名。`,
        formula_cleared: '公式聚焦已清空，当前重新显示全量结果。',
        calculating: '后端正在计算中...',
        calculation_done: '计算完成。现在你可以点选任意公式，递归查看计算过程和代码。',
        chip_bank_empty: '计算完成后，这里会出现可点击的公式列表。',
        active_bank_empty: '当前未聚焦任何公式，下面显示的是全量结果。',
        summary_variable_count: '变量数',
        summary_assignment_count: '基础组合数',
        summary_total_probability: '输入概率和',
        summary_normalized: '已归一化',
        yes: '是',
        no: '否',
        states_inline: (states) => `states: ${states}`,
        nonzero_distribution: '非零联合分布预览',
        rows_label: (count) => `${count} 项`,
        formula_results_empty: '当前没有可展示的公式结果。',
        explanation_empty: '聚焦某个公式后，这里会递归展示它依赖的计算过程。',
        code_empty: '聚焦公式后，这里会展示对应的项目根目录 Python 计算代码。',
        all_results_empty: '这里会显示全部熵、条件熵、互信息等结果。',
        direct_from_joint: '直接由联合分布计算',
        focus_target: '聚焦目标',
        dependency_item: '依赖项',
        explanation_title: '递归计算过程',
        explanation_steps: (count) => `${count} 步`,
        process_label: '计算式',
        dependency_label: '依赖',
        code_title: '代码计算过程',
        code_formula_count: (count) => `${count} 个公式`,
        copy_code: '复制代码',
        copied: '代码已复制到剪贴板。',
        copy_failed: '复制失败，请手动复制。',
        group_entropies: '熵',
        group_conditional_entropies: '条件熵',
        group_mutual_informations: '互信息',
        group_conditional_mutual_informations: '条件互信息',
        matrix_empty: '至少需要两个随机变量才能展示二维概率表。',
        matrix_title: '二维概率分布表',
        matrix_row_axis: '纵轴变量',
        matrix_col_axis: '横轴变量',
        matrix_overlap_error: '横纵轴变量不能重叠，请重新选择。',
        matrix_prob_header: 'P',
    },
    en: {
        hero_title: 'pyShannon Information Theory Calculator',
        hero_text: 'A web interface driven directly by the existing Python computation core. Define random variables, derived variables, and joint distributions, then inspect entropy, conditional entropy, mutual information, and recursive formula dependencies with code output.',
        workspace_title: 'Variable Workspace',
        workspace_text: 'Define base random variables first, then add derived variables when needed.',
        base_vars_title: 'Base Variables',
        derived_vars_title: 'Derived Variables',
        joint_table_title: 'Joint Probability Table',
        result_title: 'Results',
        result_text: 'All values are computed by the backend Python code. The frontend handles filtering, recursive explanation, and code display.',
        add_variable: 'Add Variable',
        add_derived_variable: 'Add Derived Variable',
        build_table: 'Build Table',
        load_sample: 'Load Exam Sample',
        fill_uniform: 'Fill Uniformly',
        normalize_table: 'Normalize Table',
        normalize_before_submit: 'Normalize before calculation',
        run_calculation: 'Calculate',
        exam_hint: 'Tip: if you only want to reproduce the exam problem, load the sample first and then run the calculation.',
        formula_placeholder: 'Type formulas and add them, e.g. H(X|Y), I(X;Y|Z)',
        add_formula_focus: 'Add Focus Formula',
        clear_formula_focus: 'Clear Focus',
        base_pill: 'Base',
        derived_pill: 'Derived',
        remove: 'Remove',
        field_key: 'Key',
        field_latex: 'LaTeX',
        field_states: 'States',
        field_expression: 'Expression',
        preview: 'Preview',
        placeholder_key: 'X',
        placeholder_latex: 'X',
        placeholder_states: '0,1',
        placeholder_expression: 'X*Y',
        waiting_input: 'Waiting for input',
        need_base_var: 'Add at least one base variable first.',
        invalid_states: (key) => `Variable ${key || '?'} needs at least two states.`,
        probability_sum: (sum) => `Current probability sum: ${sum.toFixed(6)}`,
        table_rebuilt: 'The probability table has been rebuilt from the current base variables.',
        normalize_zero_error: 'The current probability sum is 0, so normalization cannot be applied.',
        normalized_ok: 'The current probability table has been normalized.',
        sample_loaded: 'The exam sample is loaded. You can calculate immediately.',
        formula_not_found: (text) => `Formula ${text} was not found. Click a formula chip or enter the exact formula name.`,
        formula_cleared: 'Focused formulas cleared. Full results are shown again.',
        calculating: 'Backend calculation in progress...',
        calculation_done: 'Calculation complete. You can now focus formulas and inspect recursive explanations and code.',
        chip_bank_empty: 'Formula chips will appear here after a calculation.',
        active_bank_empty: 'No focused formula yet. Full results are shown below.',
        summary_variable_count: 'Variables',
        summary_assignment_count: 'Base assignments',
        summary_total_probability: 'Input sum',
        summary_normalized: 'Normalized',
        yes: 'Yes',
        no: 'No',
        states_inline: (states) => `states: ${states}`,
        nonzero_distribution: 'Non-zero Joint Distribution Preview',
        rows_label: (count) => `${count} rows`,
        formula_results_empty: 'No formulas are available under the current focus.',
        explanation_empty: 'Focus a formula to recursively show its dependency chain here.',
        code_empty: 'Focus a formula to show the corresponding Python computation code here.',
        all_results_empty: 'All entropy and mutual-information results will appear here.',
        direct_from_joint: 'Computed directly from the joint distribution',
        focus_target: 'Focused formula',
        dependency_item: 'Dependency',
        explanation_title: 'Recursive Calculation Trace',
        explanation_steps: (count) => `${count} steps`,
        process_label: 'Formula',
        dependency_label: 'Depends on',
        code_title: 'Generated Code',
        code_formula_count: (count) => `${count} formulas`,
        copy_code: 'Copy Code',
        copied: 'Code copied to clipboard.',
        copy_failed: 'Copy failed. Please copy it manually.',
        group_entropies: 'Entropy',
        group_conditional_entropies: 'Conditional Entropy',
        group_mutual_informations: 'Mutual Information',
        group_conditional_mutual_informations: 'Conditional Mutual Information',
        matrix_empty: 'At least two random variables are required to show a 2D probability table.',
        matrix_title: '2D Probability Table',
        matrix_row_axis: 'Row axis',
        matrix_col_axis: 'Column axis',
        matrix_overlap_error: 'Row and column axes must be disjoint.',
        matrix_prob_header: 'P',
    },
};

function t(key, ...args) {
    const value = I18N[currentLang][key];
    if (typeof value === 'function') return value(...args);
    return value ?? key;
}

function typesetMath() {
    if (window.MathJax && window.MathJax.typesetPromise) {
        window.MathJax.typesetPromise().catch(() => {});
    }
}

function applyStaticTranslations() {
    document.documentElement.lang = currentLang === 'zh' ? 'zh-CN' : 'en';
    document.querySelectorAll('[data-i18n]').forEach((el) => {
        el.innerHTML = t(el.dataset.i18n);
    });
    document.querySelectorAll('[data-i18n-placeholder]').forEach((el) => {
        el.placeholder = t(el.dataset.i18nPlaceholder);
    });
    langSwitch.textContent = currentLang === 'zh' ? 'EN' : '中';
    document.title = t('hero_title');
}

function setStatus(message, kind = 'neutral') {
    statusBanner.textContent = message;
    statusBanner.className = `status-banner ${kind}`;
}

function nextUniqueKey(baseName, existingKeys) {
    if (!existingKeys.has(baseName)) return baseName;
    let i = 1;
    while (existingKeys.has(`${baseName}${i}`)) i += 1;
    return `${baseName}${i}`;
}

function parseStates(rawValue) {
    return rawValue.split(',').map((item) => item.trim()).filter(Boolean);
}

function compactJoin(tokens) {
    if (tokens.every((token) => /^[A-Za-z0-9]$/.test(token))) return tokens.join('');
    return tokens.join(',');
}

function updateVariablePreview(card) {
    const key = card.querySelector('.var-key').value.trim() || '?';
    const latexInput = card.querySelector('.var-latex');
    const latex = latexInput ? (latexInput.value.trim() || key) : key;
    card.querySelector('.latex-preview').textContent = `\\(${latex}\\)`;
    typesetMath();
}

function attachCardEvents(card) {
    card.querySelectorAll('input').forEach((input) => input.addEventListener('input', () => updateVariablePreview(card)));
    card.querySelector('.remove-btn').addEventListener('click', () => {
        card.remove();
        buildProbabilityTable();
        applyStaticTranslations();
    });
    updateVariablePreview(card);
}

function addBaseVariable(data = {}) {
    const card = document.getElementById('base-variable-template').content.firstElementChild.cloneNode(true);
    card.querySelector('.var-key').value = data.key || '';
    card.querySelector('.var-latex').value = data.latex || data.key || '';
    card.querySelector('.var-states').value = data.states || '0,1';
    attachCardEvents(card);
    baseList.appendChild(card);
    applyStaticTranslations();
}

function addDerivedVariable(data = {}) {
    const card = document.getElementById('derived-variable-template').content.firstElementChild.cloneNode(true);
    card.querySelector('.var-key').value = data.key || '';
    card.querySelector('.var-latex').value = data.latex || data.key || '';
    card.querySelector('.var-expression').value = data.expression || '';
    attachCardEvents(card);
    derivedList.appendChild(card);
    applyStaticTranslations();
}

function getBaseVariables() {
    return [...baseList.querySelectorAll('.base-card')].map((card) => ({
        key: card.querySelector('.var-key').value.trim(),
        latex: card.querySelector('.var-latex').value.trim(),
        states: parseStates(card.querySelector('.var-states').value.trim()),
    }));
}

function getDerivedVariables() {
    return [...derivedList.querySelectorAll('.derived-card')].map((card) => ({
        key: card.querySelector('.var-key').value.trim(),
        latex: card.querySelector('.var-latex').value.trim(),
        expression: card.querySelector('.var-expression').value.trim(),
    })).filter((item) => item.key || item.expression || item.latex);
}

function cartesianProduct(arrays) {
    return arrays.reduce((acc, current) => acc.flatMap((prefix) => current.map((item) => [...prefix, item])), [[]]);
}

function currentProbabilityMap() {
    const map = new Map();
    probabilityWrap.querySelectorAll('tr[data-assignment]').forEach((row) => {
        map.set(row.dataset.assignment, row.querySelector('.probability-input').value);
    });
    return map;
}

function updateProbabilitySum() {
    let sum = 0;
    probabilityWrap.querySelectorAll('.probability-input').forEach((input) => {
        const value = Number(input.value);
        if (!Number.isNaN(value)) sum += value;
    });
    sumBanner.textContent = t('probability_sum', sum);
}

function buildProbabilityTable(probabilitySeed = null) {
    const baseVars = getBaseVariables().filter((item) => item.key);
    if (!baseVars.length) {
        probabilityWrap.className = 'table-wrap empty-state';
        probabilityWrap.textContent = t('need_base_var');
        updateProbabilitySum();
        return;
    }

    const invalidStateVar = baseVars.find((item) => item.states.length < 2);
    if (invalidStateVar) {
        probabilityWrap.className = 'table-wrap empty-state';
        probabilityWrap.textContent = t('invalid_states', invalidStateVar.key || '?');
        updateProbabilitySum();
        return;
    }

    const preserved = probabilitySeed || currentProbabilityMap();
    const table = document.createElement('table');
    table.className = 'prob-table';

    const thead = document.createElement('thead');
    const headRow = document.createElement('tr');
    baseVars.forEach((item) => {
        const th = document.createElement('th');
        th.textContent = item.key;
        headRow.appendChild(th);
    });
    const probHead = document.createElement('th');
    probHead.textContent = t('matrix_prob_header');
    headRow.appendChild(probHead);
    thead.appendChild(headRow);
    table.appendChild(thead);

    const tbody = document.createElement('tbody');
    cartesianProduct(baseVars.map((item) => item.states)).forEach((labels) => {
        const assignment = {};
        labels.forEach((label, idx) => {
            assignment[baseVars[idx].key] = label;
        });
        const signature = JSON.stringify(assignment);
        const row = document.createElement('tr');
        row.dataset.assignment = signature;
        labels.forEach((label) => {
            const td = document.createElement('td');
            td.textContent = label;
            row.appendChild(td);
        });
        const td = document.createElement('td');
        const input = document.createElement('input');
        input.type = 'number';
        input.step = 'any';
        input.className = 'probability-input';
        input.value = preserved.get ? (preserved.get(signature) || '0') : (preserved[signature] || '0');
        input.addEventListener('input', updateProbabilitySum);
        td.appendChild(input);
        row.appendChild(td);
        tbody.appendChild(row);
    });

    table.appendChild(tbody);
    probabilityWrap.className = 'table-wrap';
    probabilityWrap.innerHTML = '';
    probabilityWrap.appendChild(table);
    updateProbabilitySum();
}

function fillUniform() {
    const inputs = [...probabilityWrap.querySelectorAll('.probability-input')];
    if (!inputs.length) return;
    const value = 1 / inputs.length;
    inputs.forEach((input) => {
        input.value = value.toFixed(6);
    });
    updateProbabilitySum();
}

function normalizeTable() {
    const inputs = [...probabilityWrap.querySelectorAll('.probability-input')];
    if (!inputs.length) return;
    const values = inputs.map((input) => Number(input.value) || 0);
    const sum = values.reduce((acc, item) => acc + item, 0);
    if (sum <= 0) {
        setStatus(t('normalize_zero_error'), 'error');
        return;
    }
    inputs.forEach((input, idx) => {
        input.value = (values[idx] / sum).toFixed(6);
    });
    updateProbabilitySum();
    setStatus(t('normalized_ok'), 'success');
}

function populateFromPayload(payload) {
    baseList.innerHTML = '';
    derivedList.innerHTML = '';
    payload.base_variables.forEach((item) => addBaseVariable({ key: item.key, latex: item.latex, states: (item.states || []).join(',') }));
    (payload.derived_variables || []).forEach((item) => addDerivedVariable({ key: item.key, latex: item.latex, expression: item.expression }));
    const seed = new Map();
    (payload.joint_probabilities || []).forEach((entry) => seed.set(JSON.stringify(entry.assignment), String(entry.probability)));
    normalizeBeforeSubmit.checked = Boolean(payload.normalize_probabilities);
    buildProbabilityTable(seed);
}

async function loadSample() {
    const res = await fetch('/api/example/problem-xyz');
    const data = await res.json();
    if (!data.ok) throw new Error(data.error || 'Unable to load sample.');
    selectedFormulas.clear();
    populateFromPayload(data.payload);
    clearResultAreas();
    setStatus(t('sample_loaded'), 'success');
}

function collectPayload() {
    if (!probabilityWrap.querySelector('.probability-input')) buildProbabilityTable();
    return {
        base_variables: getBaseVariables(),
        derived_variables: getDerivedVariables(),
        joint_probabilities: [...probabilityWrap.querySelectorAll('tr[data-assignment]')].map((row) => ({
            assignment: JSON.parse(row.dataset.assignment),
            probability: row.querySelector('.probability-input').value,
        })),
        normalize_probabilities: normalizeBeforeSubmit.checked,
    };
}

function renderSummary(summary) {
    const cards = [
        [t('summary_variable_count'), summary.variable_count],
        [t('summary_assignment_count'), summary.base_assignment_count],
        [t('summary_total_probability'), Number(summary.total_probability).toFixed(6)],
        [t('summary_normalized'), summary.normalized_input ? t('yes') : t('no')],
    ];
    summaryCards.innerHTML = cards.map(([label, value]) => `
        <div class="summary-card">
            <span class="label">${label}</span>
            <span class="value">${value}</span>
        </div>
    `).join('');
}

function renderVariableSummary(variables) {
    variableSummary.innerHTML = variables.map((item) => `
        <div class="variable-badge">
            <div class="metric-formula">\\(${item.latex}\\)</div>
            <div class="states-inline">${t('states_inline', item.state_labels.join(', '))}</div>
        </div>
    `).join('');
}

function renderDistribution(rows, variables) {
    if (!rows.length) {
        distributionPreview.innerHTML = '<div class="empty-state">0</div>';
        return;
    }
    const head = variables.map((item) => `<th>${item.key}</th>`).join('');
    const body = rows.map((row) => {
        const cells = variables.map((item) => `<td>${row.assignment[item.key]}</td>`).join('');
        return `<tr>${cells}<td>${Number(row.probability).toFixed(6)}</td></tr>`;
    }).join('');
    distributionPreview.innerHTML = `
        <div class="result-group">
            <div class="result-group-head">
                <h3>${t('nonzero_distribution')}</h3>
                <span>${t('rows_label', rows.length)}</span>
            </div>
            <div class="table-wrap">
                <table class="distribution-table">
                    <thead><tr>${head}<th>${t('matrix_prob_header')}</th></tr></thead>
                    <tbody>${body}</tbody>
                </table>
            </div>
        </div>
    `;
}

function getAllRecords(result) {
    return result.formula_order.map((formula) => result.formula_index[formula]);
}

function renderMetricGroup(title, records) {
    if (!records.length) return '';
    const cards = records.map((item) => `
        <button type="button" class="metric-card metric-button ${selectedFormulas.has(item.formula) ? 'selected' : ''}" data-select-formula="${item.formula}">
            <div class="metric-formula">\\(${item.latex}\\)</div>
            <div class="metric-value">${Number(item.value).toFixed(6)}</div>
        </button>
    `).join('');
    return `
        <section class="result-group">
            <div class="result-group-head">
                <h3>${title}</h3>
                <span>${records.length}</span>
            </div>
            <div class="metric-grid">${cards}</div>
        </section>
    `;
}

function buildCategoryGroups(records) {
    const bucket = { entropies: [], conditional_entropies: [], mutual_informations: [], conditional_mutual_informations: [] };
    records.forEach((record) => bucket[record.category].push(record));
    return bucket;
}

function renderFormulaChipBank(result) {
    if (!result || !result.formula_order.length) {
        formulaChipBank.className = 'chip-bank empty-state';
        formulaChipBank.innerHTML = t('chip_bank_empty');
        activeFormulaBank.innerHTML = '';
        return;
    }
    formulaChipBank.className = 'chip-bank';
    formulaChipBank.innerHTML = result.formula_order.map((formula) => {
        const record = result.formula_index[formula];
        return `<button type="button" class="formula-pick ${selectedFormulas.has(formula) ? 'selected' : ''}" data-select-formula="${formula}"><span>\\(${record.latex}\\)</span></button>`;
    }).join('');
    activeFormulaBank.innerHTML = !selectedFormulas.size
        ? `<div class="empty-inline">${t('active_bank_empty')}</div>`
        : [...selectedFormulas].map((formula) => {
            const record = result.formula_index[formula];
            return `<button type="button" class="formula-pick active" data-select-formula="${formula}"><span>\\(${record.latex}\\)</span></button>`;
        }).join('');
    typesetMath();
}

function addSelectedFormula(formula) {
    if (!lastResult) return;
    const text = formula.trim();
    if (!text) return;
    if (!lastResult.formula_index[text]) {
        setStatus(t('formula_not_found', text), 'error');
        return;
    }
    selectedFormulas.add(text);
    renderResults(lastResult);
}

function toggleSelectedFormula(formula) {
    if (selectedFormulas.has(formula)) selectedFormulas.delete(formula);
    else selectedFormulas.add(formula);
    renderResults(lastResult);
}

function resolveFormulaClosure(result, formulas) {
    const visited = new Set();
    const order = [];
    function dfs(formula) {
        if (visited.has(formula)) return;
        visited.add(formula);
        const record = result.formula_index[formula];
        if (!record) return;
        record.dependencies.forEach(dfs);
        order.push(formula);
    }
    formulas.forEach(dfs);
    return order;
}

function renderExplanation(result, closureOrder) {
    if (!closureOrder.length) {
        explanationRoot.className = 'results-root empty-state';
        explanationRoot.innerHTML = t('explanation_empty');
        return;
    }
    const cards = closureOrder.map((formula, idx) => {
        const record = result.formula_index[formula];
        const deps = record.dependencies.length
            ? record.dependencies.map((dep) => {
                const depRecord = result.formula_index[dep];
                return `<button type="button" class="inline-dependency" data-select-formula="${dep}">\\(${depRecord.latex}\\)</button>`;
            }).join('')
            : `<span class="direct-note">${t('direct_from_joint')}</span>`;
        const rootFlag = selectedFormulas.has(formula)
            ? `<span class="focus-badge">${t('focus_target')}</span>`
            : `<span class="focus-badge secondary">${t('dependency_item')}</span>`;
        return `
            <article class="explain-card ${selectedFormulas.has(formula) ? 'is-root' : ''}">
                <div class="explain-head">
                    <div>
                        <div class="step-no">Step ${idx + 1}</div>
                        <div class="metric-formula">\\(${record.latex}\\)</div>
                    </div>
                    ${rootFlag}
                </div>
                <div class="metric-value">${Number(record.value).toFixed(6)}</div>
                <div class="process-line">${t('process_label')}：\\(${record.process_latex}\\)</div>
                <div class="dependency-line">${t('dependency_label')}：${deps}</div>
            </article>
        `;
    }).join('');
    explanationRoot.className = 'results-root';
    explanationRoot.innerHTML = `
        <section class="result-group">
            <div class="result-group-head">
                <h3>${t('explanation_title')}</h3>
                <span>${t('explanation_steps', closureOrder.length)}</span>
            </div>
            <div class="explain-grid">${cards}</div>
        </section>
    `;
}

function buildCodeText(result, closureOrder) {
    const parts = [result.code_prelude];
    closureOrder.forEach((formula) => {
        const record = result.formula_index[formula];
        parts.push(`# ${record.formula}`);
        parts.push(record.code);
    });
    parts.push('');
    parts.push('# Print focused formulas');
    [...selectedFormulas].forEach((formula) => {
        parts.push(`print(${JSON.stringify(formula)}, '=', results[${JSON.stringify(formula)}])`);
    });
    return parts.join('\n\n');
}

async function copyCode() {
    if (!lastResult || !selectedFormulas.size) return;
    try {
        const closureOrder = resolveFormulaClosure(lastResult, [...selectedFormulas]);
        await navigator.clipboard.writeText(buildCodeText(lastResult, closureOrder));
        setStatus(t('copied'), 'success');
    } catch (error) {
        setStatus(t('copy_failed'), 'error');
    }
}

function renderCode(result, closureOrder) {
    if (!closureOrder.length) {
        codeRoot.className = 'code-root empty-state';
        codeRoot.innerHTML = t('code_empty');
        return;
    }
    const codeText = buildCodeText(result, closureOrder);
    codeRoot.className = 'code-root';
    codeRoot.innerHTML = `
        <section class="result-group">
            <div class="result-group-head">
                <h3>${t('code_title')}</h3>
                <div class="button-row compact">
                    <span>${t('code_formula_count', closureOrder.length)}</span>
                    <button type="button" class="ghost-btn" id="copy-code-btn">${t('copy_code')}</button>
                </div>
            </div>
            <pre class="code-block"><code>${escapeHtml(codeText)}</code></pre>
        </section>
    `;
    document.getElementById('copy-code-btn').addEventListener('click', copyCode);
}

function escapeHtml(text) {
    return text.replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;');
}

function getSubsetOptions(variables) {
    const options = [];
    const indices = variables.map((_, idx) => idx);
    for (let size = 1; size <= indices.length; size += 1) {
        const choose = (start, picked) => {
            if (picked.length === size) {
                const labels = picked.map((idx) => variables[idx].key);
                options.push({ key: picked.join('-'), indices: [...picked], label: compactJoin(labels) });
                return;
            }
            for (let i = start; i < indices.length; i += 1) {
                picked.push(indices[i]);
                choose(i + 1, picked);
                picked.pop();
            }
        };
        choose(0, []);
    }
    return options;
}

function renderMatrix(result) {
    if (!result || result.variables.length < 2) {
        matrixRoot.className = 'matrix-root empty-state';
        matrixRoot.innerHTML = t('matrix_empty');
        return;
    }

    const options = getSubsetOptions(result.variables);
    if (!matrixState.rowAxis) matrixState.rowAxis = options[0].key;
    if (!matrixState.colAxis) {
        const rowIndices = matrixState.rowAxis.split('-').map(Number);
        const preferredCol = options.find((opt) => opt.indices.length === result.variables.length - rowIndices.length && !opt.indices.some((idx) => rowIndices.includes(idx)));
        const fallbackCol = options.find((opt) => !opt.indices.some((idx) => rowIndices.includes(idx)));
        matrixState.colAxis = preferredCol ? preferredCol.key : (fallbackCol ? fallbackCol.key : options[0].key);
    }

    const rowOption = options.find((opt) => opt.key === matrixState.rowAxis) || options[0];
    const colOption = options.find((opt) => opt.key === matrixState.colAxis) || options[0];
    const overlap = rowOption.indices.some((idx) => colOption.indices.includes(idx));

    const selectHtml = (id, currentKey, labelText) => `
        <label class="axis-picker">
            <span>${labelText}</span>
            <select id="${id}" class="axis-select">
                ${options.map((opt) => `<option value="${opt.key}" ${opt.key === currentKey ? 'selected' : ''}>${opt.label}</option>`).join('')}
            </select>
        </label>
    `;

    if (overlap) {
        matrixRoot.className = 'matrix-root';
        matrixRoot.innerHTML = `
            <section class="result-group">
                <div class="result-group-head"><h3>${t('matrix_title')}</h3></div>
                <div class="matrix-toolbar">${selectHtml('row-axis-select', rowOption.key, t('matrix_row_axis'))}${selectHtml('col-axis-select', colOption.key, t('matrix_col_axis'))}</div>
                <div class="empty-state">${t('matrix_overlap_error')}</div>
            </section>
        `;
        bindMatrixAxisEvents();
        return;
    }

    const rowKeys = rowOption.indices.map((idx) => result.variables[idx].key);
    const colKeys = colOption.indices.map((idx) => result.variables[idx].key);
    const rowStates = cartesianProduct(rowKeys.map((key) => result.variables.find((v) => v.key === key).state_labels));
    const colStates = cartesianProduct(colKeys.map((key) => result.variables.find((v) => v.key === key).state_labels));

    const matrix = new Map();
    result.rows.forEach((row) => {
        const rowLabel = compactJoin(rowKeys.map((key) => String(row.assignment[key])));
        const colLabel = compactJoin(colKeys.map((key) => String(row.assignment[key])));
        const key = `${rowLabel}__${colLabel}`;
        matrix.set(key, (matrix.get(key) || 0) + Number(row.probability));
    });

    const rowLabels = rowStates.map((labels) => compactJoin(labels.map(String)));
    const colLabels = colStates.map((labels) => compactJoin(labels.map(String)));
    const thead = `<tr><th>${rowOption.label} \\ ${colOption.label}</th>${colLabels.map((label) => `<th>${label}</th>`).join('')}</tr>`;
    const tbody = rowLabels.map((rowLabel) => {
        const cells = colLabels.map((colLabel) => `<td>${(matrix.get(`${rowLabel}__${colLabel}`) || 0).toFixed(6)}</td>`).join('');
        return `<tr><th>${rowLabel}</th>${cells}</tr>`;
    }).join('');

    matrixRoot.className = 'matrix-root';
    matrixRoot.innerHTML = `
        <section class="result-group">
            <div class="result-group-head"><h3>${t('matrix_title')}</h3></div>
            <div class="matrix-toolbar">${selectHtml('row-axis-select', rowOption.key, t('matrix_row_axis'))}${selectHtml('col-axis-select', colOption.key, t('matrix_col_axis'))}</div>
            <div class="table-wrap"><table class="distribution-table axis-table"><thead>${thead}</thead><tbody>${tbody}</tbody></table></div>
        </section>
    `;
    bindMatrixAxisEvents();
}

function bindMatrixAxisEvents() {
    const rowSelect = document.getElementById('row-axis-select');
    const colSelect = document.getElementById('col-axis-select');
    if (!rowSelect || !colSelect) return;
    rowSelect.addEventListener('change', () => {
        matrixState.rowAxis = rowSelect.value;
        if (lastResult) renderMatrix(lastResult);
    });
    colSelect.addEventListener('change', () => {
        matrixState.colAxis = colSelect.value;
        if (lastResult) renderMatrix(lastResult);
    });
}

function renderResultsList(result, closureOrder) {
    const records = closureOrder.length ? closureOrder.map((formula) => result.formula_index[formula]) : getAllRecords(result);
    const grouped = { entropies: [], conditional_entropies: [], mutual_informations: [], conditional_mutual_informations: [] };
    records.forEach((record) => grouped[record.category].push(record));
    const html = [
        [t('group_entropies'), grouped.entropies],
        [t('group_conditional_entropies'), grouped.conditional_entropies],
        [t('group_mutual_informations'), grouped.mutual_informations],
        [t('group_conditional_mutual_informations'), grouped.conditional_mutual_informations],
    ].map(([title, items]) => renderMetricGroup(title, items)).join('');
    resultsRoot.className = 'results-root';
    resultsRoot.innerHTML = html || `<div class="empty-state">${t('formula_results_empty')}</div>`;
}

function clearResultAreas() {
    formulaChipBank.className = 'chip-bank empty-state';
    formulaChipBank.innerHTML = t('chip_bank_empty');
    activeFormulaBank.innerHTML = '';
    summaryCards.innerHTML = '';
    variableSummary.innerHTML = '';
    distributionPreview.innerHTML = '';
    matrixRoot.className = 'matrix-root empty-state';
    matrixRoot.innerHTML = t('matrix_empty');
    explanationRoot.className = 'results-root empty-state';
    explanationRoot.innerHTML = t('explanation_empty');
    codeRoot.className = 'code-root empty-state';
    codeRoot.innerHTML = t('code_empty');
    resultsRoot.className = 'results-root empty-state';
    resultsRoot.innerHTML = t('all_results_empty');
}

function renderResults(result) {
    renderSummary(result.summary);
    renderVariableSummary(result.variables);
    renderDistribution(result.rows, result.variables);
    renderMatrix(result);
    renderFormulaChipBank(result);
    const closureOrder = selectedFormulas.size ? resolveFormulaClosure(result, [...selectedFormulas]) : [];
    renderExplanation(result, closureOrder);
    renderCode(result, closureOrder);
    renderResultsList(result, closureOrder);
    typesetMath();
}

async function runCalculation() {
    try {
        const payload = collectPayload();
        setStatus(t('calculating'), 'neutral');
        const res = await fetch('/api/calculate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
        });
        const data = await res.json();
        if (!data.ok) throw new Error(data.error || 'Calculation failed.');
        lastResult = data.result;
        renderResults(lastResult);
        setStatus(t('calculation_done'), 'success');
    } catch (error) {
        setStatus(error.message, 'error');
    }
}

function handleFormulaInput() {
    const raw = formulaInput.value.trim();
    if (!raw) return;
    raw.split(',').map((item) => item.trim()).filter(Boolean).forEach(addSelectedFormula);
    formulaInput.value = '';
}

document.addEventListener('click', (event) => {
    const formulaButton = event.target.closest('[data-select-formula]');
    if (formulaButton && lastResult) toggleSelectedFormula(formulaButton.dataset.selectFormula);
});

document.getElementById('add-base-variable').addEventListener('click', () => {
    const existing = new Set(getBaseVariables().map((item) => item.key).filter(Boolean));
    const key = nextUniqueKey('X', existing);
    addBaseVariable({ key, latex: key, states: '0,1' });
});

document.querySelectorAll('[data-quick-var]').forEach((button) => {
    button.addEventListener('click', () => {
        const existing = new Set(getBaseVariables().map((item) => item.key).filter(Boolean));
        const key = nextUniqueKey(button.dataset.quickVar, existing);
        addBaseVariable({ key, latex: key, states: '0,1' });
    });
});

document.getElementById('add-derived-variable').addEventListener('click', () => {
    const existing = new Set([...getBaseVariables().map((item) => item.key), ...getDerivedVariables().map((item) => item.key)].filter(Boolean));
    const key = nextUniqueKey('Z', existing);
    addDerivedVariable({ key, latex: key, expression: '' });
});

document.getElementById('build-table').addEventListener('click', () => {
    buildProbabilityTable();
    setStatus(t('table_rebuilt'), 'neutral');
});

document.getElementById('fill-uniform').addEventListener('click', fillUniform);
document.getElementById('normalize-table').addEventListener('click', normalizeTable);
document.getElementById('run-calculation').addEventListener('click', runCalculation);
document.getElementById('add-formula-focus').addEventListener('click', handleFormulaInput);
document.getElementById('clear-formula-focus').addEventListener('click', () => {
    selectedFormulas.clear();
    if (lastResult) {
        renderResults(lastResult);
        setStatus(t('formula_cleared'), 'neutral');
    }
});
formulaInput.addEventListener('keydown', (event) => {
    if (event.key === 'Enter') {
        event.preventDefault();
        handleFormulaInput();
    }
});

document.getElementById('load-sample').addEventListener('click', async () => {
    try {
        await loadSample();
    } catch (error) {
        setStatus(error.message, 'error');
    }
});

langSwitch.addEventListener('click', () => {
    currentLang = currentLang === 'zh' ? 'en' : 'zh';
    applyStaticTranslations();
    updateProbabilitySum();
    if (lastResult) renderResults(lastResult);
    else {
        clearResultAreas();
        setStatus(t('waiting_input'), 'neutral');
    }
});

window.addEventListener('DOMContentLoaded', () => {
    applyStaticTranslations();
    addBaseVariable({ key: 'X', latex: 'X', states: '0,1' });
    addBaseVariable({ key: 'Y', latex: 'Y', states: '0,1' });
    buildProbabilityTable();
    updateProbabilitySum();
    clearResultAreas();
    setStatus(t('waiting_input'), 'neutral');
    typesetMath();
});

