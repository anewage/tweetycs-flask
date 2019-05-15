function circularHeatChart() {
    var svg, container, arcGroup, radialLabelGroup, segmentLabelGroup;
    var blankColor = 'grey';
    blankColor = 'light-grey';
    var negativeSentimentColor =  "#ae0000";

    var positiveSentimentColor = "#7fc97f";
    positiveSentimentColor = "#409a17";
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
    range = ["white", "red"],
    offset = 200,
    accessor = function(d) {return d;},
    radialLabels = segmentLabels = [];
    var colorScale = d3.scaleLinear();
    var colorAccessor = function(d){ return d; };

    function chart(selection) {
        selection.each(function(data) {
            svg = d3.select(this);

            var colorScale = d3.scaleLinear()
                .domain([-1, 1])
                .range([negativeSentimentColor, positiveSentimentColor]);

            if(!container){
                container = svg.append("g")
                    .classed("circular-heat", true)
                    .attr("transform", "translate(" + parseInt(margin.left + offset) + "," + parseInt(margin.top + offset) + ")");

                arcGroup = container.append('g')
                    .attr('class', 'arcGroup');

//                radialLabelGroup = container.append('g').attr('class', 'radialLabelGroup');
  //              segmentLabelGroup = container.append('g').attr('class', 'segmentLabelGroup');
            }

            var numElements = data.length;
            var autoDomain = false;
            if (domain === null) {
            }
            if(autoDomain)
                domain = null;

            //NEED TO CHANGE TO ENTER UPDATE AND EXIT
            arcPaths = arcGroup.selectAll("path").data(data);

            arcPaths
                .enter().append("path")
                .attr("d", d3.arc().innerRadius(ir).outerRadius(or).startAngle(sa).endAngle(ea))
                //.attr("fill", function(d) {return colorAccessor(d);})
                .attr('fill', function(d){
                    if (d.netSentimentValue < 0)
                        return negativeSentimentColor;
                    else
                        return positiveSentimentColor;
                   })
              //  .attr("id", function(d, i) { return (i+0) % numSegments == 0 ? "radial-label-path-"+refineString(d.category) : null;})
                .filter(function(d){ return d.isEmpty; })
                .attr('fill', blankColor)
                //.style('stroke', blankColor
                // .style('stroke', blankColor)
                //.style('stroke-width', '2px');

            arcPaths//.transition()
                .attr("d", d3.arc().innerRadius(ir).outerRadius(or).startAngle(sa).endAngle(ea))
                .attr('fill', function(d){
                    if(d.netSentimentValue < 0)
                        return negativeSentimentColor;
                    else
                        return positiveSentimentColor;
                })
                .filter(function(d){ return d.isEmpty; })
                .attr('fill', blankColor)
                //.style('stroke', blankColor)
                //.style('stroke-width', '1px');

            arcPaths.exit().remove();
/*

            var radialLabelsGroup = radialLabelGroup
                .selectAll('.radialLabels')
                .data(radialLabels);

            radialLabelsGroup
                .enter().append("text")
                .attr('class', 'radialLabels')
                .attr('dy', function(d, i) { return 10; })
                .append("textPath")
                .attr("xlink:href", function(d) {return "#radial-label-path-"+refineString(d);})
                .attr('startOffset', '50%')
                .style("font-size", 0.6 * segmentHeight + 'px')
                .style("fill", 'black')
            //    .style('text-anchor', 'middle')
                .text(function(d) {return d.substr(0,7);});


            //so we need to create a label one that is just specific for the label and shonws on the side.
            //also create a specific one that the top as well.
            //they shoould be called label Methods
//http://www.visualcinnamon.com/2015/09/placing-text-on-arcs.html
            var segmentLabelsGroup = segmentLabelGroup
                .selectAll('.segmentLabels')
                .data(segmentLabels);

            segmentLabelsGroup
                .enter().append("text")
                .attr('class', 'segmentLabels')
                .append("textPath")
                .attr("xlink:href", function(d) {return "#segment-label-path-"+refineString(d);})
                .style("font-size", 0.6 * segmentHeight + 'px')
                .style('text-anchor', 'middle')
                .text(function(d) {return d;});


            */
        });

    }

    /* Arc functions */
    ir = function(d, i) {
    //    console.log('ir' + innerRadius + Math.floor(i/numSegments) * segmentHeight);
        return innerRadius + Math.floor(i/numSegments) * segmentHeight;
    };
    or = function(d, i) {
        //.log( innerRadius + segmentHeight + Math.floor(i/numSegments) * segmentHeight  - stackSpace)
        return innerRadius + segmentHeight + Math.floor(i/numSegments) * segmentHeight  - stackSpace;
    };
    /*
    *   sa = function(d, i) {
     return (i * 2 * Math.PI) / numSegments;
     }
     ea = function(d, i) {
     return ((i + 1) * 2 * Math.PI) / numSegments;
     }
     */
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