const input = document.getElementById('query_text')
const output1 = document.getElementById('resultBox1')
const output2 = document.getElementById('resultBox2')
const output3 = document.getElementById('resultBox3')
const section1 = document.getElementById('vector_section')
const section2 = document.getElementById('keyword_section')
const section3 = document.getElementById('hybrid_section')
const filterButton = document.getElementById('filterBtn')


async function doVectorSearch() {
    section1.style.display = 'block';
    output1.textContent = '';
    output1.innerHTML = '<img src="static/images/Loading_icon.gif">'
    const response = await fetch('/api/vector_search?q=' + input.value);
    if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
    }
    const result = await response.json();
    output1.innerHTML = JSON.stringify(result, null, 2);
}

async function doKeywordSearch() {
    section2.style.display = 'block';
    output2.textContent = '';
    output2.innerHTML = '<img src="static/images/Loading_icon.gif">'
    const response = await fetch('/api/keyword_search?q=' + input.value);
    if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
    }
    const result = await response.json();
    output2.innerHTML = JSON.stringify(result, null, 2);
}

async function doHybridSearch() {
    section3.style.display = 'block';
    output3.textContent = '';
    output3.innerHTML = '<img src="static/images/Loading_icon.gif">';
    const response = await fetch('/api/hybrid_search?q=' + input.value);
    if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
    }
    const result = await response.json();
    output3.innerHTML = JSON.stringify(result, null, 2);
    filterButton.disabled = result.results.length <= 3
}

async function filterHybrid() {
    console.log('filterButton clicked');
    output3.textContent = '';
    output3.innerHTML = '<img src="static/images/Loading_icon.gif">';
    const response = await fetch('/api/hybrid_search_top_k?q=' + input.value);
    if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
    }
    const result = await response.json();
    output3.innerHTML = JSON.stringify(result, null, 2);
    filterButton.disabled = true
}

if (filterButton) {
    filterButton.addEventListener('click', filterHybrid);
} else {
    console.error('filterButton not found!');
}


clearBtn.addEventListener('click', function() {
    input.value = '';
    input.focus();
    section1.style.display = 'none'
    section2.style.display = 'none'
    section3.style.display = 'none'
});

