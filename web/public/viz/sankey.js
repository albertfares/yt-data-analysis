
// Sankey Diagram Logic - Fetching Data from JSON

(function () {
    console.log('Sankey script initializing...');

    // Configuration
    // Try absolute path for local development if relative fails
    const DATA_URL = 'sankey_data.json';
    const CONTAINER_ID = 'sankeyPlot';
    const SELECT_ID = 'sankeyChannelSelect';

    // State
    let channelData = null;
    let allCategories = null;

    // DOM Elements
    const channelSelect = document.getElementById(SELECT_ID);

    if (!channelSelect) {
        console.error(`Sankey: Element with ID '${SELECT_ID}' not found!`);
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
            console.log('Sankey: Data loaded successfully');
            channelData = data.channelData;
            allCategories = data.allCategories;

            initializeDropdown();

            // Add event listener
            channelSelect.addEventListener('change', updateSankey);

            // Set default to VICE if available
            if (channelData['VICE']) {
                channelSelect.value = 'VICE';
            }

            // Initial render
            updateSankey();
        })
        .catch(error => {
            console.error('Sankey: Error loading data:', error);
            showErrorState('Failed to load data. Please try refreshing the page.');
        });

    function initializeDropdown() {
        if (!channelData) return;

        const channels = Object.keys(channelData).sort();
        console.log(`Sankey: Populating dropdown with ${channels.length} channels`);

        channels.forEach(channel => {
            const option = document.createElement('option');
            option.value = channel;
            option.textContent = channel;
            channelSelect.appendChild(option);
        });
    }

    function showErrorState(message) {
        Plotly.newPlot(CONTAINER_ID, [], {
            title: {
                text: message,
                font: { size: 16, color: '#ff6b6b' }
            },
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)'
        });
    }

    function updateSankey() {
        const channel = channelSelect.value;
        console.log(`Sankey: Update called for channel: ${channel}`);

        if (!channel) {
            console.log('Sankey: No channel selected');
            // Show empty state
            Plotly.newPlot(CONTAINER_ID, [], {
                title: {
                    text: 'Select a channel to see the discovery flow',
                    font: { size: 16, color: '#ffffff' }
                },
                paper_bgcolor: 'rgba(0,0,0,0)',
                plot_bgcolor: 'rgba(0,0,0,0)'
            });
            return;
        }

        if (!channelData) {
            console.error('Sankey: channelData is null!');
            return;
        }

        if (!channelData[channel]) {
            console.error(`Sankey: No data found in map for channel ${channel}`);
            return;
        }

        const data = channelData[channel];
        console.log(`Sankey: Found data for ${channel}`, data);

        if (!data.paths || Object.keys(data.paths).length === 0) {
            Plotly.newPlot(CONTAINER_ID, [], {
                title: {
                    text: `No path data available for <b>${channel}</b>`,
                    font: { size: 16, color: '#ffffff' }
                },
                paper_bgcolor: 'rgba(0,0,0,0)',
                plot_bgcolor: 'rgba(0,0,0,0)',
                annotations: [{
                    text: 'This channel might not have connections in the network.',
                    showarrow: false,
                    xref: 'paper',
                    yref: 'paper',
                    x: 0.5,
                    y: 0.5,
                    font: { size: 14, color: '#aaa' }
                }]
            });
            return;
        }

        // Build Sankey Data
        const nodes = [];
        const nodeMap = {};
        const links = [];
        let nodeIdx = 0;

        // Helper to get or create node
        function getOrCreateNode(group, isSource = false, isTarget = false) {
            // Remove _TARGET suffix if present
            const cleanGroup = group.replace('_TARGET', '');
            const parts = cleanGroup.split('|');
            const cat = parts[0];
            const ch = parts.length > 1 ? parts[1] : '';
            const key = group; // Keep original key with _TARGET for uniqueness

            if (!(key in nodeMap)) {
                let label, color;

                if (isSource) {
                    // Source nodes: just show category
                    label = cat;
                    color = window.getCategoryColor ? window.getCategoryColor(cat) : '#E53935';
                } else if (isTarget) {
                    // Target nodes: show category with channel in parentheses
                    label = `${cat}\n(${ch})`;
                    color = window.getCategoryColor ? window.getCategoryColor(cat) : '#FFFFFF';
                } else {
                    // Intermediate nodes: show category with channel in parentheses
                    // Keep red as it represents the channel connection flow?
                    // User said "color per category". But intermediate is channels.
                    // Let's keep intermediates as Bright Red to distinguish flow, OR use the category color?
                    // The prompt said "stick with those colors for everytime we talk about categories".
                    // Intermediary nodes are technically (Category | Channel).
                    // If we color them by category, it might look nice.
                    label = `${cat}\n(${ch})`;
                    // color = window.getCategoryColor ? window.getCategoryColor(cat) : '#FF5252';
                    // Actually, let's keep intermediate distinct so the flow is visible. 
                    // Or follow the instruction strictly. "Everytime we talk about categories".
                    // The intermediate node IS a category context.
                    // However, if source and intermediate comprise the same category, they merge visually if colors are same.
                    // Let's stick to the previous design for intermediate (Bright Red) to show it's the "active path",
                    // OR maybe color it by the category but slightly lighter?
                    // For now, I will interpret "one color per category" as applying primarily to the main category nodes.
                    // BUT, the sankey is mostly categories.
                    // Let's try applying the category color to intermediate nodes too, maybe with some opacity or just solid.
                    color = window.getCategoryColor ? window.getCategoryColor(cat) : '#FF5252';
                }

                nodes.push({ label, color });
                nodeMap[key] = nodeIdx++;
            }

            return nodeMap[key];
        }

        // For each target category, find THE shortest path across ALL source categories
        const shortestPathsToTargets = {};

        // Collect all target categories
        const allTargetCats = new Set();
        for (const sourceCat in data.paths) {
            for (const targetCat in data.paths[sourceCat]) {
                allTargetCats.add(targetCat);
            }
        }

        // For each target category, find the globally shortest path
        for (const targetCat of allTargetCats) {
            let shortestPath = null;
            let shortestLength = Infinity;

            // Check paths from all source categories
            for (const sourceCat in data.paths) {
                const path = data.paths[sourceCat][targetCat];
                if (path && path.length > 0 && path.length < shortestLength) {
                    shortestPath = path;
                    shortestLength = path.length;
                }
            }

            if (shortestPath) {
                shortestPathsToTargets[targetCat] = shortestPath;
            }
        }

        // Now process only the shortest paths
        const pathCounts = {};

        for (const targetCat in shortestPathsToTargets) {
            const path = shortestPathsToTargets[targetCat];

            // Special handling for 1-hop paths to ensure target appears on the right
            // Special handling for 1-hop paths: Direct connection Source → Target
            if (path.length === 2) {
                const sourceGroup = path[0];
                const targetGroup = path[1];

                const sourceIdx = getOrCreateNode(sourceGroup, true, false);
                const targetIdx = getOrCreateNode(targetGroup + '_TARGET', false, true); // Green target

                // Add single direct link
                const linkKey = `${sourceIdx}-${targetIdx}`;
                pathCounts[linkKey] = (pathCounts[linkKey] || 0) + 1;
            } else {
                // For longer paths: Process normally
                for (let i = 0; i < path.length - 1; i++) {
                    const fromGroup = path[i];
                    const toGroup = path[i + 1];

                    const isSourceNode = (i === 0);
                    const isLastNode = (i + 1 === path.length - 1);

                    const fromIdx = getOrCreateNode(fromGroup, isSourceNode, false);
                    // Last node is the target
                    const toIdx = getOrCreateNode(toGroup + (isLastNode ? '_TARGET' : ''), false, isLastNode);

                    const linkKey = `${fromIdx}-${toIdx}`;
                    pathCounts[linkKey] = (pathCounts[linkKey] || 0) + 1;
                }
            }
        }

        // Convert pathCounts to links array
        for (const linkKey in pathCounts) {
            const [source, target] = linkKey.split('-').map(Number);
            links.push({
                source,
                target,
                value: pathCounts[linkKey],
                color: 'rgba(229, 57, 53, 0.2)' // Semi-transparent red
            });
        }

        if (nodes.length === 0 || links.length === 0) {
            Plotly.newPlot(CONTAINER_ID, [], {
                title: {
                    text: `No connections found for <b>${channel}</b>`,
                    font: { size: 16, color: '#ffffff' }
                },
                paper_bgcolor: 'rgba(0,0,0,0)',
                plot_bgcolor: 'rgba(0,0,0,0)'
            });
            return;
        }

        const trace = {
            type: 'sankey',
            orientation: 'h',
            node: {
                pad: 15,
                thickness: 20,
                line: {
                    color: 'rgba(255,255,255,0.5)',
                    width: 1
                },
                label: nodes.map(n => n.label),
                color: nodes.map(n => n.color),
                font: {
                    color: 'white',
                    size: 10
                }
            },
            link: {
                source: links.map(l => l.source),
                target: links.map(l => l.target),
                value: links.map(l => l.value),
                color: links.map(l => l.color)
            }
        };

        const layout = {
            title: {
                text: `Discovery Flow: <b>${channel}</b>`,
                font: { size: 16, color: '#ffffff' }
            },
            font: { size: 12, color: '#ffffff' },
            showlegend: false, // Hide legend
            height: 600,
            margin: { l: 20, r: 20, t: 60, b: 20 },
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)'
        };

        Plotly.newPlot(CONTAINER_ID, [trace], layout, {
            responsive: true,
            displayModeBar: false
        });
    }

})();
