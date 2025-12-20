
(function () {
    console.log("Category Network script initializing...");

    const CONTAINER_ID = 'categoryNetworkPlot';
    const DATA_URL = 'category_network.json';

    fetch(DATA_URL)
        .then(response => {
            if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
            return response.json();
        })
        .then(data => {
            console.log("Category Network data loaded", data);
            renderNetwork(data);
        })
        .catch(error => {
            console.error("Error loading category network data:", error);
            document.getElementById(CONTAINER_ID).innerHTML = `<div style="color:white; text-align:center; padding:20px;">Error loading graph: ${error.message}</div>`;
        });

    function renderNetwork(data) {
        const nodes = data.nodes;
        const edges = data.edges;

        const traces = [];

        // 1. Edges (Lines)
        // We create a single trace for all edges using 'None' to separate lines if possible, 
        // OR individual traces if we need variable width. 
        // Plotly scatter lines allow valid variable width in recent versions? 
        // No, standard scatter lines usually have fixed width per trace. 
        // To have variable width, we might need multiple traces or a different approach (like shapes or creating 'bins' of width).
        // Given typically limited number of categories (<20?), individual traces might be okay, or grouping by width.
        // Let's try grouping edges into a few width buckets for performance if needed, 
        // but for <100 edges, separate traces is fine. 

        // Actually, for a clean look with variable width, individual Scatter traces for edges is the most robust way in standard Plotly.js without WebGL if count is low.
        // If count is high, this is slow. 
        // Let's check edge count. ~15-20 categories -> ~20 edges. Individual traces are totally fine.

        edges.forEach(edge => {
            // Inject multiple points along the edge to ensure continuous hover detection
            // This interpolates "invisible" data points along the line
            const x0 = edge.x[0];
            const x1 = edge.x[1];
            const y0 = edge.y[0];
            const y1 = edge.y[1];

            const numSteps = 20; // Number of interpolation points
            const xVals = [];
            const yVals = [];

            for (let i = 0; i <= numSteps; i++) {
                xVals.push(x0 + (x1 - x0) * (i / numSteps));
                yVals.push(y0 + (y1 - y0) * (i / numSteps));
            }

            traces.push({
                x: xVals,
                y: yVals,
                mode: 'lines',
                line: {
                    width: Math.max(edge.width * 1.5, 4), // Much thicker for easier hovering
                    color: 'rgba(255, 255, 255, 0.5)' // More visible
                },
                hovertemplate: `Shared Commentators: ${parseInt(edge.weight).toLocaleString()}<extra></extra>`,
                showlegend: false,
                type: 'scatter'
            });
        });

        // 2. Nodes (Scatter points)
        const nodeX = nodes.map(n => n.x);
        const nodeY = nodes.map(n => n.y);
        const nodeSizes = nodes.map(n => n.size);
        const nodeTexts = nodes.map(n => `${n.name}<br>${n.count_formatted} commentators`);
        const nodeColors = nodes.map(n => window.getCategoryColor ? window.getCategoryColor(n.name) : '#E53935');

        traces.push({
            x: nodeX,
            y: nodeY,
            mode: 'markers+text',
            marker: {
                size: nodeSizes,
                color: nodeColors,
                opacity: 1, // Explicitly opaque
                line: {
                    color: 'white',
                    width: 2
                }
            },
            text: nodes.map(n => n.name),
            textposition: 'top center',
            textfont: {
                family: 'sans-serif',
                size: 14,
                color: 'white'
            },
            hoverinfo: 'text',
            hovertext: nodeTexts,
            name: 'Categories',
            type: 'scatter'
        });

        const layout = {
            showlegend: false,
            hovermode: 'closest',
            margin: { l: 20, r: 20, t: 60, b: 20 },
            xaxis: {
                showgrid: false,
                zeroline: false,
                showticklabels: false,
                fixedrange: true // Disable zoom/pan for fixed layout
            },
            yaxis: {
                showgrid: false,
                zeroline: false,
                showticklabels: false,
                fixedrange: true
            },
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            height: 600
        };

        const config = {
            responsive: true,
            displayModeBar: false
        };

        Plotly.newPlot(CONTAINER_ID, traces, layout, config);
    }
})();
