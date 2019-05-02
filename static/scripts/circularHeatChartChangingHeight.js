function circularHeatChartV() {
    var svg, container, arcGroup, axisGroup;
    var blankColor = 'grey';
    var negativeSentimentColor =  "#ae0000";
    var negativeSentimentColor = '#A9A9A9';
    var positiveSentimentColor = "#7fc97f";
    positiveSentimentColor = "#409a17";
    var showAxis = false;
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
    offset = 0,
    accessor = function(d) {return d;},
    radialLabels = segmentLabels = [];
    var colorScale = d3.scaleLinear();
    var colorAccessor = function(d){ return d; };
    var heightScale = d3.scaleLinear();
    function chart(selection) {
        selection.each(function (data) {


            heightScale =  d3.scaleLinear()
                .domain([0, -1]).range([0, innerRadius]);
                //.domain([-1, 0]).range([0, segmentHeight]);

            var svg = d3.select(this);
            var colorScale = d3.scaleLinear()
                .domain([-1, 1])
                .range([negativeSentimentColor, positiveSentimentColor]);

            if (!container) {
                container = svg.append("g")
                    .classed("circular-heat", true)
                    .attr("transform", "translate(" + parseInt(margin.left + offset) + "," + parseInt(margin.top + offset) + ")");

                arcGroup = container.append('g').attr('class', 'arcGroup');
                axisGroup = container.append('g').attr('class', 'axisGroup')
                    .attr('transform', 'translate(480,400)rotate(42)');
            //.attr('transform', 'translate(500,400)rotate(40)');

            }

            arcPaths = arcGroup.selectAll("path").data(data);

            arcPaths
                .enter()
                .append("path")
                .attr('class', 'variableArc')
                .attr("d", d3.arc().innerRadius(ir).outerRadius(or).startAngle(sa).endAngle(ea))
             //   .attr('stroke-width', '5px')
                .attr('fill', function (d) {
                    if (d.netSentimentValue < 0)
                        return negativeSentimentColor;
                    else
                        return positiveSentimentColor;
                })
                //.attr('fill', 'none')
                .filter(function (d) {  return d.isEmpty;     })
                .attr('fill', 'none')
                //.style('stroke', blankColor)
                //.style('stroke-width', '2px');

            arcPaths//.transition()
                .attr("d", d3.arc().innerRadius(ir).outerRadius(or).startAngle(sa).endAngle(ea))
                .attr('fill', function (d) {
                    if (d.netSentimentValue < 0)
                        return negativeSentimentColor;
                    else
                        return positiveSentimentColor;
                })
                .filter(function (d) {return d.isEmpty;})
                .attr('fill', 'none')
                //.style('stroke', blankColor)
                //.style('stroke-width', '1px');

            arcPaths.exit().remove();

        if (showAxis){
            var axisScale =  d3.scaleLinear()
              //  .domain([-1, 0]).range([0, segmentHeight+140])
                .domain([0, -1]).range([0, innerRadius*0.75]);

            var xAxis = d3.axisBottom(axisScale)
                .ticks(3);

            axisGroup.call(xAxis);

            axisGroup.append('text')
                .attr('x', 100)
                .attr('y', -25)
                .attr('fill', "#777")
                .text('Net sentiment rate')
        }


        })
    }

    /* Arc functions */
    ir = function(d, i) {
           // return innerRadius + (Math.floor(i/numSegments) * segmentHeight);// - 6*(20 - 10*Math.random());
      //  console.log(innerRadius)
        //console.log(heightScale(accessor(d)))
        return heightScale(accessor(d)) + innerRadius;
    //    return innerRadius + heightScale(accessor(d));
        //    return innerRadius + (Math.floor(i/numSegments) * segmentHeight)- heightScale(accessor(d));
    };
    or = function(d, i) {
            return innerRadius + segmentHeight + Math.floor(i/numSegments) * segmentHeight  - stackSpace;
    };
    sa = function(d, i) {
        return startRadians + ((i%numSegments)  * (endRadians-startRadians)) / numSegments;
    };
    ea = function(d, i) {
        if ( i % 12 == 0){
         return startRadians + (((i%numSegments) + 1)  * (endRadians-startRadians)) / numSegments   - arcSpace/1;
     }
        return startRadians + (((i%numSegments) + 1)  * (endRadians-startRadians)) / numSegments   - arcSpace;
    };

    /* Configuration getters/setters */


    chart.showAxis = function(_) {
        if (!arguments.length) return showAxis;
        showAxis = _;
        return chart;
    };

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
