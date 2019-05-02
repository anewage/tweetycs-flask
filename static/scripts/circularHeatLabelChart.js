function circularHeatChartLabel() {
    var svg, container, arcGroup, radialLabelGroup;
    var margin = {top: 20, right: 20, bottom: 20, left: 20},
    innerRadius = 50,
    numSegments = 24,
    segmentHeight = 20,
    stackSpace = 1,
    arcSpace = 0.01,
    domain = null,
    hasVariableHeight = false,
    startRadians = 0,
    endRadians = 2*Math.PI/3,
    offset = 200,
    accessor = function(d) {return d;},
    radialLabels = segmentLabels = [];

    function chart(selection) {
        selection.each(function(data) {
            svg = d3.select(this);

            if(!container){
                container = svg.append("g")
                    .attr("transform", "translate(" + parseInt(margin.left + offset) + "," + parseInt(margin.top + offset) + ")");
                arcGroup = container.append('g').attr('class', 'arcGroup');
                radialLabelGroup = container.append('g').attr('class', 'radialLabelGroup');
            }

            var numElements = data.length;

            //NEED TO CHANGE TO ENTER UPDATE AND EXIT
            arcPaths = arcGroup.selectAll("path").data(data);

            arcPaths
                .enter().append("path")
                .attr("d", d3.arc().innerRadius(ir).outerRadius(or).startAngle(sa).endAngle(ea))
                .attr("id", function(d, i) {
                    return (i+0) % numSegments == 0 ? "radial-label-path-"+refineString(d.category) : null;
                })
                .attr('fill', 'none')
                .attr('stroke', 'none');
                //.style('stroke-width', '2px');

            arcPaths//.transition()
                .attr("d", d3.arc().innerRadius(ir).outerRadius(or).startAngle(sa).endAngle(ea))
                .attr("id", function(d, i) {
                    return (i+0) % numSegments == 0 ? "radial-label-path-"+refineString(d.category) : null;
                })
                //.attr('stroke', 'black')
                //.style('stroke-width', '2px');


            arcPaths.exit().remove();


            var radialLabelsGroup = radialLabelGroup
                .selectAll('.radialLabels')
                .data(radialLabels);

            radialLabelsGroup
                .enter().append("text")
                .attr('class', 'radialLabels')
                .attr('dy', 10)
                .append("textPath")
                .attr("xlink:href", function(d) {return "#radial-label-path-"+refineString(d);})
                .attr('startOffset', function(d) {
                    //this needs to be by the length of the text and the font, use the other function you have for GEo
                    return 75 - (d.length - 10) + '%'  ; })
                .style("font-size", 0.8 * segmentHeight + 'px')
                .style("fill", 'black')
                .style('text-anchor', 'middle')
                .text(function(d) {return d;});



        });

    }

    /* Arc functions */
    ir = function(d, i) {
        return innerRadius + Math.floor(i/numSegments) * segmentHeight;
    };
    or = function(d, i) {
        return innerRadius + segmentHeight + Math.floor(i/numSegments) * segmentHeight  - stackSpace;
    };
    sa = function(d, i) {
        return startRadians + ((i%numSegments)  * (endRadians-startRadians)) / numSegments;
    };
    ea = function(d, i) {
        return startRadians + (((i%numSegments) + 1)  * (endRadians-startRadians)) / numSegments   - arcSpace;
    };

    /* Configuration getters/setters */

    chart.offset = function(_) {
        if (!arguments.length) return offset;
        offset = _;
        return chart;
    };

    chart.startRadians = function(_) {
        if (!arguments.length) return startRadians;
        startRadians = _;
        return chart;
    };
    chart.endRadians = function(_) {
        if (!arguments.length) return endRadians;
        endRadians = _;
        return chart;
    };
    chart.margin = function(_) {
        if (!arguments.length) return margin;
        margin = _;
        return chart;
    };

    chart.innerRadius = function(_) {
        if (!arguments.length) return innerRadius;
        innerRadius = _;
        return chart;
    };

    chart.hasVariableHeight = function(_) {
        if (!arguments.length) return hasVariableHeight;
        hasVariableHeight = _;
        return chart;
    };

    chart.numSegments = function(_) {
        if (!arguments.length) return numSegments;
        numSegments = _;
        return chart;
    };

    chart.segmentHeight = function(_) {
        if (!arguments.length) return segmentHeight;
        segmentHeight = _;
        return chart;
    };

    chart.domain = function(_) {
        if (!arguments.length) return domain;
        domain = _;
        return chart;
    };

    chart.range = function(_) {
        if (!arguments.length) return range;
        range = _;
        return chart;
    };

    chart.arcSpace = function(_) {
        if (!arguments.length) return arcSpace;
        arcSpace = _;
        return chart;
    };
    chart.stackSpace = function(_) {
        if (!arguments.length) return stackSpace;
        stackSpace= _;
        return chart;
    };

    chart.radialLabels = function(_) {
        if (!arguments.length) return radialLabels;
        if (_ == null) _ = [];
        radialLabels = _;
        return chart;
    };

    chart.segmentLabels = function(_) {
        if (!arguments.length) return segmentLabels;
        if (_ == null) _ = [];
        segmentLabels = _;
        return chart;
    };

    chart.accessor = function(_) {
        if (!arguments.length) return accessor;
        accessor = _;
        return chart;
    };

    chart.colorScale = function(_) {
        if (!arguments.length) return colorScale;
        colorScale = _;
        return chart;
    };

    chart.colorAccessor = function(_) {
        if (!arguments.length) return colorAccessor;
        colorAccessor = _;
        return chart;
    };

    return chart;
}




function refineString (name) {
    return name.replace(/\s+/g, '').replace(/\&/g, '').replace(/\'/g, '').replace(/\(/g,'').replace(/\)/g,'').replace(/\//g,'').replace(/\,/g,''); //remove white space, ',' '(' and ')'
}