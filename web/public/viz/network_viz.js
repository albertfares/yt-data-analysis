// Network Visualization using D3.js
// Visualizes the real "Hub Video" structure extracted from analysis

document.addEventListener('DOMContentLoaded', function () {
    const container = document.getElementById('network-container');
    const width = container.clientWidth;
    const height = container.clientHeight;

    // Helper function to parse CSV text
    function parseCSV(csvText) {
        const lines = csvText.trim().split('\n');
        const headers = lines[0].split(',');
        return lines.slice(1).map(line => {
            const values = line.split(',');
            return headers.reduce((obj, header, index) => {
                obj[header] = values[index];
                return obj;
            }, {});
        });
    }

    async function initNetworkViz() {
        try {
            // Fetch Nodes and Edges
            const [nodesRes, edgesRes] = await Promise.all([
                fetch('data/network_ego_FCjiMVHbXlQ_nodes.csv'),
                fetch('data/network_ego_FCjiMVHbXlQ_edges.csv')
            ]);

            const nodesText = await nodesRes.text();
            const edgesText = await edgesRes.text();

            const nodesData = parseCSV(nodesText);
            const edgesData = parseCSV(edgesText);

            // Process Nodes
            const nodes = nodesData.map(d => ({
                id: d.video_id,
                label: d.type === 'hub' ? 'Harlem Shake FAIL (Hub)' : `Video ${d.video_id.substring(0, 5)}...`,
                group: d.type,
                radius: d.type === 'hub' ? 25 : 5 + Math.sqrt(parseInt(d.degree || 1)), // Size by degree
                color: d.type === 'hub' ? '#FFFFFF' : '#FF3B30' // White for hub, Red for neighbors
            }));

            // Process Links
            const links = edgesData.map(d => ({
                source: d.source,
                target: d.target,
                value: parseFloat(d.weight || 1),
                overlap: parseInt(d.overlap || 1)
            }));

            // Setup D3 Simulation
            const svg = d3.select("#network-container")
                .append("svg")
                .attr("width", width)
                .attr("height", height)
                .attr("viewBox", [0, 0, width, height])
                .style("max-width", "100%")
                .style("height", "auto");

            const simulation = d3.forceSimulation(nodes)
                .force("link", d3.forceLink(links).id(d => d.id).distance(d => d.source.id === "FCjiMVHbXlQ" ? 150 : 50))
                .force("charge", d3.forceManyBody().strength(-300))
                .force("center", d3.forceCenter(width / 2, height / 2))
                .force("collide", d3.forceCollide().radius(d => d.radius + 5));

            // Render Elements

            // Links
            const link = svg.append("g")
                .attr("stroke", "rgba(255, 255, 255, 0.2)")
                .attr("stroke-opacity", 0.6)
                .selectAll("line")
                .data(links)
                .join("line")
                .attr("stroke-width", d => Math.sqrt(d.value));

            // Nodes
            const node = svg.append("g")
                .attr("stroke", "#fff")
                .attr("stroke-width", 1.5)
                .selectAll("circle")
                .data(nodes)
                .join("circle")
                .attr("r", d => d.radius)
                .attr("fill", d => d.color)
                .attr("fill-opacity", 0.8)
                .call(drag(simulation));

            // Labels (only for Hub)
            const label = svg.append("g")
                .attr("class", "labels")
                .selectAll("text")
                .data(nodes.filter(d => d.group === 'hub'))
                .join("text")
                .attr("dx", 30)
                .attr("dy", 5)
                .text(d => d.label)
                .style("fill", "white")
                .style("font-family", "system-ui")
                .style("font-weight", "bold")
                .style("text-shadow", "0 2px 4px rgba(0,0,0,0.8)");

            // Tooltip
            const tooltip = d3.select("body").append("div")
                .attr("class", "d3-tooltip")
                .style("position", "absolute")
                .style("visibility", "hidden")
                .style("background", "rgba(0, 0, 0, 0.8)")
                .style("color", "#fff")
                .style("padding", "8px")
                .style("border-radius", "4px")
                .style("font-family", "system-ui")
                .style("font-size", "12px")
                .style("pointer-events", "none")
                .style("backdrop-filter", "blur(4px)");

            // Interactions
            node.on("mouseover", function (event, d) {
                d3.select(this).attr("stroke", "#FFD700").attr("stroke-width", 3);
                tooltip.style("visibility", "visible")
                    .html(`<strong>${d.label}</strong><br>Type: ${d.group}`);
            })
                .on("mousemove", function (event) {
                    tooltip.style("top", (event.pageY - 10) + "px")
                        .style("left", (event.pageX + 10) + "px");
                })
                .on("mouseout", function () {
                    d3.select(this).attr("stroke", "#fff").attr("stroke-width", 1.5);
                    tooltip.style("visibility", "hidden");
                });

            // Simulation Tick
            simulation.on("tick", () => {
                link
                    .attr("x1", d => d.source.x)
                    .attr("y1", d => d.source.y)
                    .attr("x2", d => d.target.x)
                    .attr("y2", d => d.target.y);

                node
                    .attr("cx", d => d.x)
                    .attr("cy", d => d.y);

                label
                    .attr("x", d => d.x)
                    .attr("y", d => d.y);
            });

            // Drag Behavior
            function drag(simulation) {
                function dragstarted(event) {
                    if (!event.active) simulation.alphaTarget(0.3).restart();
                    event.subject.fx = event.subject.x;
                    event.subject.fy = event.subject.y;
                }

                function dragged(event) {
                    event.subject.fx = event.x;
                    event.subject.fy = event.y;
                }

                function dragended(event) {
                    if (!event.active) simulation.alphaTarget(0);
                    event.subject.fx = null;
                    event.subject.fy = null;
                }

                return d3.drag()
                    .on("start", dragstarted)
                    .on("drag", dragged)
                    .on("end", dragended);
            }

        } catch (error) {
            console.error("Error loading network data:", error);
        }
    }

    initNetworkViz();
});
