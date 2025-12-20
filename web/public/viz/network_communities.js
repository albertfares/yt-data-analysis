// Network Communities Visualization

(function () {
    console.log('Network communities script initializing...');

    // Configuration
    const DATA_URL = 'network_communities.json';
    const CONTAINER_ID = 'networkPlot';
    const SELECT_ID = 'communitySelect';
    const SLIDER_ID = 'nodeSlider';
    const COUNT_ID = 'nodeCount';

    // State
    let communityData = null;
    let currentCommunity = null;
    let currentNodeCount = 50;

    // DOM Elements
    const communitySelect = document.getElementById(SELECT_ID);
    const nodeSlider = document.getElementById(SLIDER_ID);
    const nodeCount = document.getElementById(COUNT_ID);

    if (!communitySelect || !nodeSlider || !nodeCount) {
        console.error('Network: Required elements not found!');
        return;
    }

    // Fetch Data
    fetch(DATA_URL)
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            console.log('Network: Data loaded successfully');
            communityData = data;
            initializeDropdown();

            // Event listeners
            communitySelect.addEventListener('change', onCommunityChange);
            nodeSlider.addEventListener('input', onSliderChange);

            // Set initial state
            if (!communitySelect.value) {
                showPlaceholder();
            }
        })
        .catch(error => {
            console.error('Network: Error loading data:', error);
            showError('Failed to load network data. Please refresh the page.');
        });

    function initializeDropdown() {
        if (!communityData) return;

        console.log(`Network: Populating dropdown with ${communityData.length} communities`);

        communityData.forEach(comm => {
            const option = document.createElement('option');
            option.value = comm.id;
            option.textContent = `Community ${comm.id}`;
            communitySelect.appendChild(option);
        });

        // Set community 1 as default
        if (communityData.length > 0) {
            communitySelect.value = '1';
            onCommunityChange();
        }
    }

    function onCommunityChange() {
        const commId = parseInt(communitySelect.value);
        if (!commId) {
            showPlaceholder();
            return;
        }

        currentCommunity = communityData.find(c => c.id === commId);

        // Update slider max
        nodeSlider.max = currentCommunity.loaded_nodes;
        nodeSlider.value = Math.min(50, currentCommunity.loaded_nodes);
        currentNodeCount = parseInt(nodeSlider.value);
        nodeCount.textContent = currentNodeCount;

        updateGraph();
        updateInfo();
    }

    function onSliderChange() {
        currentNodeCount = parseInt(nodeSlider.value);
        nodeCount.textContent = currentNodeCount;
        if (currentCommunity) {
            updateGraph();
        }
    }

    function showPlaceholder() {
        Plotly.newPlot(CONTAINER_ID, [], {
            title: {
                text: 'Select a community to explore the network',
                font: { size: 16, color: 'rgba(255, 255, 255, 0.7)' }
            },
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            xaxis: { visible: false },
            yaxis: { visible: false }
        });

        // Hide info
        const infoBox = document.getElementById('networkInfo');
        if (infoBox) infoBox.style.display = 'none';
    }

    function showError(message) {
        Plotly.newPlot(CONTAINER_ID, [], {
            title: {
                text: message,
                font: { size: 16, color: '#ff6b6b' }
            },
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)'
        });
    }

    function updateGraph() {
        if (!currentCommunity) return;

        // Filter nodes
        const visibleNodes = currentCommunity.nodes.slice(0, currentNodeCount);
        const visibleNodeIds = new Set(visibleNodes.map(n => n.id));

        // Filter edges - check if both endpoints are in visible nodes by position
        const visibleEdges = currentCommunity.edges.filter(e => {
            return visibleNodes.some(n => n.x === e.x0 && n.y === e.y0) &&
                visibleNodes.some(n => n.x === e.x1 && n.y === e.y1);
        });

        // Prepare edge traces
        const edgeX = [];
        const edgeY = [];
        visibleEdges.forEach(e => {
            edgeX.push(e.x0, e.x1, null);
            edgeY.push(e.y0, e.y1, null);
        });

        const edgeTrace = {
            x: edgeX,
            y: edgeY,
            mode: 'lines',
            line: { width: 0.5, color: 'rgba(255, 255, 255, 0.15)' },
            hoverinfo: 'none',
            showlegend: false
        };

        // Prepare node trace
        const nodeX = visibleNodes.map(n => n.x);
        const nodeY = visibleNodes.map(n => n.y);
        const nodeText = visibleNodes.map(n => `<b>${n.channel}</b><br>${n.category}<br>Degree: ${n.degree}`);
        // Use consistent category colors
        const nodeColors = visibleNodes.map(n => window.getCategoryColor ? window.getCategoryColor(n.category) : '#999');
        const nodeSizes = visibleNodes.map(n => n.size);

        const nodeTrace = {
            x: nodeX,
            y: nodeY,
            mode: 'markers',
            hoverinfo: 'text',
            text: nodeText,
            marker: {
                color: nodeColors,
                size: nodeSizes,
                opacity: 0.9,
                line: { width: 1, color: 'rgba(255, 255, 255, 0.5)' }
            },
            showlegend: false
        };

        // Label top nodes
        const labelsToShow = Math.min(15, currentNodeCount);
        const labelTrace = {
            x: visibleNodes.slice(0, labelsToShow).map(n => n.x),
            y: visibleNodes.slice(0, labelsToShow).map(n => n.y),
            mode: 'text',
            text: visibleNodes.slice(0, labelsToShow).map(n => n.channel.slice(0, 15)),
            textposition: 'top center',
            textfont: { size: 9, color: 'rgba(255, 255, 255, 0.8)' },
            hoverinfo: 'none',
            showlegend: false
        };

        const data = [edgeTrace, nodeTrace, labelTrace];

        const layout = {
            title: {
                text: `<b>Community ${currentCommunity.id}</b>`,
                font: { size: 20, color: 'rgba(255, 255, 255, 0.9)' },
                x: 0.5,
                xanchor: 'center'
            },
            showlegend: false,
            hovermode: 'closest',
            margin: { b: 20, l: 20, r: 80, t: 60 },
            xaxis: {
                showgrid: false,
                zeroline: false,
                showticklabels: false,
                showline: false
            },
            yaxis: {
                showgrid: false,
                zeroline: false,
                showticklabels: false,
                showline: false
            },
            plot_bgcolor: 'rgba(0, 0, 0, 0)',
            paper_bgcolor: 'rgba(0, 0, 0, 0)',
            font: { color: 'rgba(255, 255, 255, 0.9)' },
            dragmode: 'pan'
        };

        const config = {
            scrollZoom: true,
            displayModeBar: true,
            modeBarButtonsToRemove: ['lasso2d', 'select2d', 'toImage'],
            displaylogo: false,
            responsive: true
        };

        Plotly.react(CONTAINER_ID, data, layout, config);

    }

    function updateInfo() {
        if (!currentCommunity) return;

        const infoBox = document.getElementById('networkInfo');
        if (infoBox) infoBox.style.display = 'block';

        const statsElem = document.getElementById('communityStats');
        if (statsElem) {
            // "Connections" here refers to the edges in the visualized subgraph (loaded_edges)
            // or we could show currentCommunity.edges.length
            statsElem.innerHTML = `
                <strong>Total Channels:</strong> ${currentCommunity.total_nodes.toLocaleString()} &nbsp;•&nbsp; 
                <strong>Connections:</strong> ${currentCommunity.edges.length.toLocaleString()}
            `;
        }

        const topChannelsElem = document.getElementById('topChannels');
        if (topChannelsElem) {
            const topChannelsHTML = currentCommunity.top_channels
                .slice(0, 12) // Limit to top 12 channels to save space
                .map((ch, i) => `
                    <li>
                        <span class="channel-rank">#${i + 1}</span>
                        <div class="channel-info">
                            <span class="channel-name-txt">${ch[0]}</span>
                            <div class="channel-degree">${ch[1]} connections</div>
                        </div>
                    </li>
                `)
                .join('');
            topChannelsElem.innerHTML = topChannelsHTML;
        }

        const topCategoriesElem = document.getElementById('topCategories');
        if (topCategoriesElem && currentCommunity.category_counts) {
            // Use pre-calculated category counts from JSON (based on ALL nodes, not just visible ones)
            const catCounts = currentCommunity.category_counts;

            // Convert to array and sort
            const sortedCats = Object.entries(catCounts)
                .sort((a, b) => b[1] - a[1])
                .slice(0, 3); // Limit to top 3 categories as requested

            // Use total_nodes from metadata for accurate percentage
            const totalNodes = currentCommunity.total_nodes;

            const catsHTML = sortedCats.map(([cat, count]) => {
                const percentage = ((count / totalNodes) * 100).toFixed(1);
                const color = window.getCategoryColor ? window.getCategoryColor(cat) : '#999';
                return `
                    <div class="category-item">
                        <div class="category-color-dot" style="background-color: ${color}"></div>
                        <span class="category-name">${cat}</span>
                        <span class="category-percentage">${count.toLocaleString()} (${percentage}%)</span>
                    </div>
                `;
            }).join('');

            topCategoriesElem.innerHTML = catsHTML;
        }
    }

})();

