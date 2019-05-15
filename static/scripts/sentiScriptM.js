if (!d3.uwo) {d3.uwo = {};}


d3.uwo.sentiViz = function module() {



    function chart(selection) {
        selection.each(function(data) {

            var atHigherLevel = data.atHigherLevel;
            var selectedCluster = "Neoplasms";
            var netPerGroup = data.netPerGroup;
            var curvedHeatMapData = data.curvedHeatMapData;
            var dataAgg = data.dataAgg;
            var dataKeywords = data.dataKeywords;
            var extraSpace = 620;
            var margin = {left:120, top:40, right:50, bottom:50};
//RESPONSIVE
//var width = Math.max( Math.min(window.innerWidth, 1800) - margin.left - margin.right - 20, 500),
//height = Math.max( Math.min(window.innerHeight - 250, 1600) - margin.top - margin.bottom - 20, 500),

            var catOrder = [
                "promotional",
                "fundraising",
                "educational",
                "personal",
                "media",
                "celebrities",
                "interest groups",
                "official agencies",
                'public',
                'businesses'
            ];

            var width = 1600 - margin.left - margin.right - 20,
                height = 1400 - margin.top - margin.bottom - 20,
                innerRadius = Math.min(width * 0.33, height * .45) *0.9,
                outerRadius = innerRadius * 1.05;

//Recalculate the width and height now that we know the radius
            width = outerRadius * 2 + margin.right + margin.left;
            height = outerRadius * 2 + margin.top + margin.bottom;

//Reset the overall font size
            var newFontSize = Math.min(70, Math.max(40, innerRadius * 62.5 / 250));
            d3.select("html").style("font-size", newFontSize + "%");

////////////////////////////////////////////////////////////
////////////////// Set-up Chord parameters /////////////////
////////////////////////////////////////////////////////////
            //20
            var pullOutSize = 110 + 30/135 * innerRadius; //you might want to change this ratio
            var numFormat = d3.format(",.0f");
            var defaultOpacity = 0.85,
                fadeOpacity = 0.075;

            svg = d3.select(this)
                .append("svg")
                .attr("id", "theactual")
                .attr("width", width + margin.left + margin.right+extraSpace*2)
                .attr("height", height + margin.top + margin.bottom+extraSpace*2);

            var loom = d3.uwo.loom()
                .padAngle(0.05)
                //.sortSubgroups(sortAlpha)
                //.heightInner(28)
                .emptyPerc(0.35)  //this is what we use to control how must space is at the top and bottom
                .widthOffsetInner(160)  //this controls how close the ribbons are to the text in the center, smaller i== closer nneds to be based on the rect size/text size
                //.widthOffsetInner(function(d) { return 6 * d.length; })
                .value(function(d) { return d.words; })
                .inner(function(d) { return d.character; })
                .outer(function(d) { return d.location; });

//this is for the arc
            var arc = d3.arc()
                .innerRadius(innerRadius*1.01) //outerRadius = innerRadius * 1.05  the space between the ribbons and the arc at the top
                .outerRadius(outerRadius);

            var string = d3.uwo.string()
                .radius(innerRadius) //radius of the ribbon circle
                .pullout(pullOutSize);  //this is for the ribbons and it controls the width of the center object

////////////////////////////////////////////////////////////
//////////////////// Character notes ///////////////////////
////////////////////////////////////////////////////////////

            // var characterNotes = [];
            // characterNotes["Gandalf"] = "Speaking almost twice as many words as the second most abundant speaker, Gandalf is taking up a large portion of dialogue in almost every location he's in, but stays rather quiet in Mordor";
            // characterNotes["Sam"] = "An unexpected runner up to having spoken the most words, Sam flourishes after the battle at Amon Hen, taking up a considerable portion of the words said in both Mordor and Gondor";
            // characterNotes["Aragorn"] = "Although eventually being crowned in Minas Tirith, Gondor, Aragorn is by far most talkative in that other human region, Rohan, fighting a battle at Helm's Deep and convincing an army of dead";
            // characterNotes["Frodo"] = "Frodo seems most comfortable speaking in the Shire, (mostly) when still an innocent youth, but he feels the burden of the ring increasingly towards the end and leaves the talking to his best friend Sam";
            // characterNotes["Gimli"] = "Gimli is a quiet character at practically all locations until he reaches Rohan, where he speaks almost half of all his lines";
            // characterNotes["Pippin"] = "Like Merry, Pippin is also seen saying something at all locations, but his presence is mostly felt when he sings his song in Minas Tirith, serving the steward of Gondor, Denethor";
            // characterNotes["Merry"] = "Merry manages to say an average sentence worth of words at all locations, but is most active during his time with Treebeard in Fangorn forest and bonding with Eowyn in Rohan";
            // characterNotes["Boromir"] = "Boromir speaks his finest lines during the march up Caradhras in the Misty Mountains and right before the Uruk-hai battle at Amon Hen, Parth Galen, taking up a large portion of the total number of words spoken at those locations";
            // characterNotes["Legolas"] = "Although a very memorable presence throughout the movies, Legolas speaks even less in 3 movies than Boromir, who is alive in only the first movie";

////////////////////// Create SVG //////////////////////////


            var heatMapGroup = svg.append('g')
                .attr("transform", "translate(" + (width/2 + margin.left +extraSpace/2) + "," + (height/2 + margin.top+extraSpace/2+80) + ")");


            var loc1 = heatMapGroup.append('g').attr('class', 'heatMapTweetGroup');
            var loc2 = heatMapGroup.append('g').attr('class', 'heatMapUserGroup');
            var loc3 = heatMapGroup.append('g').attr('class', 'heatMapOverallGroup');
            var loc4 = heatMapGroup.append('g').attr('class', 'heatMapTweetLabelGroup');
            var loc5 = heatMapGroup.append('g').attr('class', 'heatMapUserLabelGroup');
            var loc6 = heatMapGroup.append('g').attr('class', 'heatMapCauseGroup')

      //      curvedHeatMapData = createHeatMapDataForEachCluster(data);
         //   console.log(curvedHeatMapData);

          //  drawBottom(processCurvedHeatMapData(curvedHeatMapData[selectedCluster] ), atHigherLevel[selectedCluster]);
    //        netSentimentByCategory(data);
       //     console.log(netPerGroup);
            drawBottomDefault(netPerGroup);
/*
            var dataChange = d3v4.nest()
                .key(function(d){ return d.cluster;})
                .key(function(d){ return d.user_category})
                .rollup(function(leaves) { return leaves.length; })
                .entries(data);


            var dataChange2 = d3v4.nest()
                .key(function(d){ return d.cluster;})
                .key(function(d){ return d.tweet_category})
                .rollup(function(leaves) { return leaves.length; })
                .entries(data);

            var totalData = dataChange.concat(dataChange2);

            var dataAgg = [];
            totalData.forEach(function(d){

                d.values.forEach(function(c){
                    var obj = {};
                    obj.character = d.key;
                    obj.location = c.key;
                    obj.words =  c.value;
                    dataAgg.push(obj);

                })

            });
            console.log(dataAgg);
            */

            drawDumbo(svg, dataAgg);

            function drawDumbo(svg, dataAgg){

                //Find the total number of words per character
                var dataChar = d3.nest()
                    .key(function(d) { return d.character; })
                    .rollup(function(leaves) { return d3.sum(leaves, function(d) { return d.words; }); })
                    .entries(dataAgg)
                    .sort(function(a, b){ return d3.descending(a.value, b.value); });
                //Unflatten the result
                var characterOrder = dataChar.map(function(d) { return d.key; });


                //Sort the characters on a specific order
                function sortCharacter(a, b) {
                    return characterOrder.indexOf(a) - characterOrder.indexOf(b);
                }//sortCharacter


                function sortCategory(a,b){
                    return catOrder.indexOf(a) - catOrder.indexOf(b);
                }

                //Set more loom functions
                loom
                    .sortSubgroups(sortCharacter)
                    .sortGroups(sortCategory)
                    //0.75  //1.90
                    .heightInner(innerRadius*2.05/characterOrder.length);  //this is the height of the labels in the center.

                ///////////////////////// Colors ///////////////////////////

                //Color for the unique locations
                var locations = [
                    "official agencies",
                    "celebrities",
                    "media",
                    "businesses",
                    "interest groups",
                    "public",

                    "educational",
                    "personal",
                    "promotional",
                    "fundraising"
                ];
                var colors = [
                    "#5a3511",
                    "#47635f",
                    "#223e15",
                    "#C6CAC9",
                    "#0d1e25",
                    "#53821a",
                    "#4387AA",
                    "#770000",
                    "#373F41",
                    "#602317",
                    "#8D9413",
                    "#c17924",
                    "#3C7E16"];
                //var colors = ["#373F41", "#373F41","#373F41","#373F41","#373F41","#373F41", "#373F41", "#602317",     "#8D9413",   "#c17924", "#3C7E16"];
                //var colors = ["#373F41", "#373F41","#373F41","#373F41","#373F41","#373F41", "#373F41",  "#c17924","#c17924","#c17924","#c17924","#c17924","#c17924","#c17924"];

                //take to the other side the bottom    "#7fc97f", "#770000", //ALT

//need four colors Pink, Purple, Orange and Blue

                var colors = [
                    "#242b39",  //make deeper
                    "#c08d60",  //make deeper
                    "#1c3956",
                    "#5a3511",
                    "#dddd7a",   //"#ffff99",
                    "#a898be",//  "#beaed4",


                    "#d38924",  //make one shade lighter
                    "#3a8ad0",   //  "#386cb0",  //"#4387AA",
                    "#770000",
                    "#6a51a3"

                ];
                var color = d3.scaleOrdinal()
                    .domain(locations)
                    .range(colors);

                //Create a group that already holds the data
                var g = svg.append("g")
                    .attr("transform", "translate(" + (width/2 + margin.left +extraSpace/2) + "," + (height/2 + margin.top+extraSpace/2) + ")")
                    .datum(loom(dataAgg));

                var titles = g.append("g")
                    .attr("class", "texts")
                    .style("opacity", 0);

                titles.append("text")
                    .attr("class", "name-title")
                    .attr("x", 0)
                    .attr("y", -innerRadius-100);

                titles.append("text")
                    .attr("class", "click-title")
                    .attr("x", 0)
                    .attr("y", innerRadius+30);

                titles.append("text")
                    .attr("class", "value-title")
                    .attr("x", 0)
                    .attr("y", -innerRadius*5/6 + 25);

                //The character pieces
                titles.append("text")
                    .attr("class", "character-note")
                    .attr("x", 0)
                    .attr("y", innerRadius/2)
                    .attr("dy", "0.35em");

                var arcs = g.append("g")
                    .attr("class", "arcs")
                    .selectAll("g")
                    .data(function(s) {
                        return s.groups;
                    })
                    .enter().append("g")
                    .attr("class", "arc-wrapper")
                    .each(function(d) {
                        d.pullOutSize = (pullOutSize * ( d.startAngle > Math.PI + 1e-2 ? -1 : 1))
                    })
                    .on("mouseover", function(d) {

                        d3.selectAll(".inner-label-rect")
                            .transition()
                            .attr("opacity", function(s) {
                                return s.name !== d.outername ? 0 : 1;
                            });

                        //Hide all other arcs
                        d3.selectAll(".arc-wrapper")
                            .transition()
                            .style("opacity", function(s) { return s.outername === d.outername ? 1 : 0.5; });

                        //Hide all other strings
                        d3.selectAll(".string")
                            .transition()
                            .style("opacity", function(s) { return s.outer.outername === d.outername ? 1 : fadeOpacity; });

                        //Find the data for the strings of the hovered over location
                        var locationData = loom(dataAgg).filter(function(s) { return s.outer.outername === d.outername; });
                        //Hide the characters who haven't said a word
                        d3.selectAll(".inner-label")
                            .transition()
                            .style("opacity", function(s) {
                                //Find out how many words the character said at the hovered over location
                                var char = locationData.filter(function(c) { return c.outer.innername === s.name; });
                                return char.length === 0 ? 0.1 : 1;
                            });
                    })
                    .on("mouseout", function(d) {

                        d3.selectAll(".inner-label-rect")
                            .transition()
                            .attr('opacity', 1);

                        //Sjow all arc labels
                        d3.selectAll(".arc-wrapper")
                            .transition()
                            .style("opacity", 1);

                        //Show all strings again
                        d3.selectAll(".string")
                            .transition()
                            .style("opacity", defaultOpacity);

                        //Show all characters again
                        d3.selectAll(".inner-label")
                            .transition()
                            .style("opacity", 1);
                    });

                var outerArcs = arcs.append("path")
                    .attr("class", "arc")
                    .style("fill", function(d) { return color(d.outername); })
                    .attr("d", arc)
                    .attr("transform", function(d, i) { //Pull the two slices apart
                        return "translate(" + d.pullOutSize + ',' + 0 + ")";
                    });

                //////////////////// Draw outer labels /////////////////////

                //The text needs to be rotated with the offset in the clockwise direction
                var outerLabels = arcs.append("g")
                    .each(function(d) { d.angle = ((d.startAngle + d.endAngle) / 2); })
                    .attr("class", "outer-labels")
                    .attr("text-anchor", function(d) { return d.angle > Math.PI ? "end" : null; })
                    .attr("transform", function(d,i) {
                        var c = arc.centroid(d);
                        return "translate(" + (c[0] + d.pullOutSize) + "," + c[1] + ")"
                            + "rotate(" + (d.angle * 180 / Math.PI - 90) + ")"
                            + "translate(" + 26 + ",0)"
                            + (d.angle > Math.PI ? "rotate(180)" : "")
                    });


                //The outer name
                outerLabels.append("text")
                    .attr("class", "outer-label")
                    .attr("dy", ".35em")
                    .text(function(d,i){ return d.outername; });



                //The value below it
                /*
                 outerLabels.append("text")
                 .attr("class", "outer-label-value")
                 .attr("dy", "1.5em")
                 .text(function(d,i){ return numFormat(d.value) + " words"; });

                 */
                ////////////////// Draw inner strings //////////////////////

                var strings = g.append("g")
                    .attr("class", "stringWrapper")
                    .style("isolation", "isolate")
                    .selectAll("path")
                    .data(function(strings) {
                        return strings;
                    })
                    .enter().append("path")
                    .attr("class", "string")
                    .style("mix-blend-mode", "multiply")
                    .attr("d", string)
                    .style("fill", function(d) { return d3.rgb( color(d.outer.outername) ).brighter(0.2) ; })
                    .style("opacity", defaultOpacity);



                //////////////////// Draw inner labels /////////////////////
                //CHANGE THIS SO IT IS A RECTANGLE BUT HOW WOULD THAT WORK :( NAMES INSIDE????

                //The text also needs to be displaced in the horizontal directions
                //And also rotated with the offset in the clockwise direction
                var innerLabels = g.append("g")
                    .attr("class","inner-labels");
                //OR MAYBE THIS SHOULD BE GOVERNED BY THE LENGTH OF THE TEXT AND NOT A FIXED RECT SIZE.


                innerLabels.selectAll('rect')
                    .data(function(s){ return s.innergroups})
                    .enter().append('rect')
                    .attr('class', "inner-label-rect")
                    //.attr('rx', 5)
                    //.attr('ry',5)
                    .attr("x", function(d){ return d.x - 161;})
                    .attr('y', function(d){return d.y - 7.5})
                    .attr('height', 15) //15
                    .attr('width',321) //280
                    .attr('fill', 'none')
                    .attr('stroke', 'none'); //none



                innerLabels.selectAll("text")
                    .data(function(s) {
                        return s.innergroups;
                    })
                    .enter().append("text")
                    .attr("class", "inner-label")
                    .attr("x", function(d,i) { return d.x; })
                    .attr("y", function(d,i) { return d.y; })
                    .style("text-anchor", "middle")
                    .attr("dy", ".35em")
                    .text(function(d) { return CAUSE_SUB_GROUP_ABBR_TWITTER[d.name]; })
                    .call(wrap,360);//260

                innerLabels.selectAll('text')
                    .on("mouseover", function(d) {

                        //Show all the strings of the highlighted character and hide all else
                        d3.selectAll(".string")
                            .transition()
                            .style("opacity", function(s) {
                                return s.outer.innername !== d.name ? fadeOpacity : 1;
                            });

                        //Update the word count of the outer labels
                        var characterData = loom(dataAgg).filter(function(s) { return s.outer.innername === d.name; });
                        d3.selectAll(".outer-label-value")
                            .text(function(s,i){
                                //Find which characterData is the correct one based on location
                                var loc = characterData.filter(function(c) { return c.outer.outername === s.outername; });
                                if(loc.length === 0) {
                                    var value = 0;
                                } else {
                                    var value = loc[0].outer.value;
                                }
                                return numFormat(value) + (value === 1 ? " word" : " words");

                            });

                        //Hide the arc where the character hasn't said a thing
                        d3.selectAll(".arc-wrapper")
                            .transition()
                            .style("opacity", function(s) {
                                //Find which characterData is the correct one based on location
                                var loc = characterData.filter(function(c) { return c.outer.outername === s.outername; });
                                return loc.length === 0 ? 0.1 : 1;
                            });

                        //Update the title to show the total word count of the character
                        d3.selectAll(".texts")
                            .transition()
                            .style("opacity", 1);

                        d3.select(".name-title")
                            .attr('dy',0)
                            .text(d.name)
                            .style("opacity", 1)
                            .call(wrap, 600); //needs to be variable

                        /*
                        d3v4.select(".value-title")
                            .text(function() {
                                var words = dataChar.filter(function(s) { return s.key === d.name; });
                                return numFormat(words[0].value);
                            });
                        */

                        //Show the character note
                        /*d3v4.selectAll(".character-note")
                            .text(characterNotes[d.name])
                            .call(wrap, 2.25*pullOutSize);

                        */
                    })
                    .on("mouseout", function(d) {

                        //Put the string opacity back to normal
                        d3.selectAll(".string")
                            .transition()
                            .style("opacity", defaultOpacity);

                        //Return the word count to what it was
                        d3.selectAll(".outer-label-value")
                            .text(function(s,i){ return numFormat(s.value) + " words"; });

                        //Show all arcs again
                        d3.selectAll(".arc-wrapper")
                            .transition()
                            .style("opacity", 1);

                        //Hide the title
/*                        d3v4.selectAll(".texts")
                            .transition()
                            .style("opacity", 0);
*/
                        d3.select(".name-title")
                            .transition()
                            .style("opacity", 0);
                    })
                    .on('click', function(d){
                        if( d.clicked == undefined )
                        {
                            d.clicked = true;

                        d3.selectAll(".inner-label-rect")
                            .transition()
                            .attr("stroke", function(s) {
                                return s.name !== d.name ? 'none' : 'black';
                            });

                        d3.select(".click-title")
                            .attr('dy',0)
                            .text(d.name)
                            .style("opacity", 1)
                            .call(wrap, 600); //needs to be variable

                        drawBottom(processCurvedHeatMapData(curvedHeatMapData[d.name] ), atHigherLevel[d.name]);
                        }
                        else {
                            d.clicked = undefined;
                            d3.selectAll(".inner-label-rect")
                                .transition()
                                .attr("stroke", 'none' )

                            d3.select(".click-title")
                                .attr('dy',0)
                                .text(d.name)
                                .style("opacity", 0)

                            drawBottomDefault(netPerGroup);
                        }

                    })
            }
            function drawPie(data, svg, radius){


                svgLocal = svg.append('g')
                    .attr('id', 'piePortion')
                    .attr("transform", "translate(" + (width/2 + margin.left + extraSpace/2) + "," + (height + margin.top+ extraSpace/20) + ")");

                var colorPie = d3.scaleOrdinal()
                    .range(["#98abc5", "#8a89a6", "#7b6888", "#6b486b", "#a05d56", "#d0743c", "#ff8c00", "#5a3511", "#47635f",   "#223e15", "#C6CAC9", "#0d1e25",  "#53821a",    "#4387AA",         "#770000", "#373F41", "#602317"]);

                var arcPie = d3.arc()
                    .outerRadius(radius - 10)
                    .innerRadius(radius - 20);

                var labelArc = d3.arc()
                    .outerRadius(radius - 0)
                    .innerRadius(radius - 0);

                var pie = d3.pie()
                    .startAngle(-2*Math.PI/3)
                    .endAngle(2*Math.PI/3)
                    .padAngle(0.015)
                    .sort(null)
                    .value(function(d) { return d.count; });
                //.value(function(d) { return d.population; });


                data = data
                    .sort(function(a,b) { return b.count - a.count});

                var g = svgLocal.selectAll(".arc")
                    .data(pie(data))
                    .enter().append("g")
                    .attr("class", "arc")
                    .on('mouseover', function(d){
                        d3.select(this).select(".arc_title").attr('opacity', "1");
                    })
                    .on('mouseout', function(d){
                        d3.select(this).select(".arc_title").attr('opacity', "0");
                    })

                g.append("path")
                    .attr("d", arcPie)
                    .style('fill',"black" )
                //.style("fill", function(d) { return colorPie(d.data.age); });


                g.append("text")
                    .attr("transform", function(d) {
                        var midAngle = d.endAngle > 0 && d.endAngle < Math.PI ? d.startAngle/2 + d.endAngle/2 : d.startAngle/2  + d.endAngle/2 + Math.PI ;
                        return "translate(" + labelArc.centroid(d)[0] + "," + labelArc.centroid(d)[1] + ") rotate(-90) rotate(" + (midAngle * 180/Math.PI) + ")"; })
                    .attr("dy", ".35em")
                    .attr('text-anchor',function(d) {
                        return   d.endAngle > 0 && d.endAngle < Math.PI ? 'start' : 'end';
                    })
                    .style('fill', 'black')
                    .text(function(d) { return d.data.keyword; });
                //.text(function(d) { return (d.data.cars > 10000) ? d.data.make : null; });

                g.append("text")
                    .attr('class', "arc_title")
                    .attr("transform", function(d) {
                        var midAngle = d.endAngle > 0 && d.endAngle < Math.PI ? d.startAngle/2 + d.endAngle/2 : d.startAngle/2  + d.endAngle/2 + Math.PI ;
                        return "translate(" + labelArc.centroid(d)[0] + "," + labelArc.centroid(d)[1] + ") rotate(-90) rotate(" + (midAngle * 180/Math.PI) + ")"; })
                    .attr("dy", ".35em")
                    .attr("y", 20)
                    .attr('text-anchor',function(d) {
                        return   d.endAngle > 0 && d.endAngle < Math.PI ? 'start' : 'end';
                    })
                    .attr('opacity', "0")
                    .style('fill', 'black')
                    .text(function(d) { return d.data.count; });

                function type(d) {
                    d.population = +d.population;
                    return d;
                }
            }
            function drawBottom(data, dataTop){
                var innerRadius = 640; //this needs to be calculated based on the other
                var level1Height = 7;
                //********************************************************************************
                loc1.selectAll('*').remove();
                loc2.selectAll('*').remove();
                loc3.selectAll('*').remove();
                loc4.selectAll('*').remove();
                loc5.selectAll('*').remove();
                loc6.selectAll('*').remove();

                var arcHeightForEachCause = 15;
                drawRegGroup(innerRadius, data[0], dataTop.length, loc1, loc4 );
                drawRegGroup(innerRadius+arcHeightForEachCause*(level1Height+2), data[1], dataTop.length, loc2, loc5);
         //       drawWeightGroup(innerRadius - 60, dataTop, loc3, arcHeightForEachCause, "netSentimentValue", false);
                drawCauseGroup((innerRadius+arcHeightForEachCause*(level1Height+7)), dataTop, false );
            }

            function drawBottomDefault(dataTop){
                var innerRadius = 640; //this needs to be calculated based on the other
                var arcHeightForEachCause = 15;
                var level1Height = 7;
                //********************************************************************************
                loc1.selectAll('*').remove();
                loc2.selectAll('*').remove();
                loc3.selectAll('*').remove();
                loc4.selectAll('*').remove();
                loc5.selectAll('*').remove();
                loc6.selectAll('*').remove();

                var arcHeightForEachCause = 120;
                drawWeightGroup(innerRadius*0.8, dataTop, loc3, arcHeightForEachCause, "netSentimentValue", true); //"netSentimentValue"
                drawCauseGroup(0.8*(innerRadius+arcHeightForEachCause+3), dataTop , true);
            }

            function drawRegGroup(radius, data, numOfSegments, container, locLabel){

                var gap = Math.PI/30;
                var arcHeightForEachCause = 15;

                var regionStartRadians = 22 * gap,
                    regionEndRadians = 2 * Math.PI - 22 * gap;

                var chart = circularHeatChart()
                    .innerRadius(radius)
                    .arcSpace(0.00) //by setting to 0 we don't know where it ends or begins
                    .stackSpace(0)
                    .numSegments(numOfSegments)
                    .segmentHeight(arcHeightForEachCause)
                    .startRadians(regionStartRadians)
                    .endRadians(regionEndRadians)
                    .margin({top: 20, right: 20, bottom: 20, left: 20});

                chart.accessor(function(d) {return d.netSentimentValue;})
                    .offset(0)
                    .radialLabels(null)
                    .segmentLabels(null);

                container.data([data]).call(chart);
                drawRegLabelGroup(radius, data, numOfSegments, locLabel)

            }
            function drawRegLabelGroup(radius, data, numOfSegments, container){

                var gap = Math.PI/30;
                var arcHeightForEachCause = 15;

                var regionStartRadians = 2 * Math.PI - 22 * gap;  //this needs to be for the length of the text.
                var regionEndRadians = regionStartRadians + 1.5*gap;

                var chart = circularHeatChartLabel()
                    .innerRadius(radius)
                    .arcSpace(0.00) //by setting to 0 we don't know where it ends or begins
                    .stackSpace(0)
                    .numSegments(1)
                    .segmentHeight(arcHeightForEachCause)
                    .startRadians(regionStartRadians)
                    .endRadians(regionEndRadians)
                    .margin({top: 20, right: 20, bottom: 20, left: 20});

                var dataToUse  = [];
                //yo should be able to use underscore to make this shorter/smaller
                data.forEach(function(d, i){
                    if (i % numOfSegments === 0 ){
                        dataToUse.push(d)
                    }
                });

                chart.accessor(function(d) {return d.netSentimentValue;})
                    .offset(0)
                    .radialLabels( dataToUse.map(function(d){ return d.category;}))
                    .segmentLabels(null);


                container.data([dataToUse]).call(chart);


            }
            function drawWeightGroup(radius, dataTop ,container, arcHeightForEachCause, access, showAxis){

                var gap = Math.PI/30;

                var regionStartRadians = 22 * gap,
                    regionEndRadians = 2 * Math.PI - 22 * gap;

                var weightChart = circularHeatChartV()
                    .innerRadius(radius)
                    .arcSpace(0.01) //by setting to 0 we don't know where it ends or begins
                    .stackSpace(1)
                    .hasVariableHeight(true)
                    .numSegments(dataTop.length)
                    .segmentHeight(arcHeightForEachCause)
                    .startRadians(regionStartRadians)
                    .endRadians(regionEndRadians)
                    .margin({top: -60, right: 20, bottom: 20, left: -30});

                weightChart.accessor(function(d) {return d[access];})
                    .offset(0)
                    .showAxis(showAxis)
                    .radialLabels(null)
                    .colorAccessor(function(d){ return d.color;})
                    .segmentLabels(null);


                var sum = d3.sum(dataTop, function(d){ return d.numTweets;});
                var dataToUse = dataTop.map(function(d) {
                    d.proportion = d.numTweets/sum*100;
                    return d;
                }).sort(function(a,b){ return d3.ascending(a.cause, b.cause); });
            //.sort(function(a,b){ return d3v4.ascending(a.netSentimentValue, b.netSentimentValue); });



                container.data([dataToUse]).call(weightChart);

            }
            function drawCauseGroup(radius, data, defaultView){

                if(defaultView)
                    loc6.attr('transform', "translate(-30,-40)");
                else
                    loc6.attr('transform', "translate(20,10)");

                var gap = Math.PI/30;

                var regionStartRadians = 22 * gap,
                    regionEndRadians = 2 * Math.PI - 22 * gap;


                var arcPie = d3.arc()
                    .outerRadius(radius)
                    .innerRadius(radius -5);

                var labelArc = d3.arc()
                    .outerRadius(radius + 5)
                    .innerRadius(radius + 15 );

                var pie = d3.pie()
                    .startAngle(regionStartRadians)
                    .endAngle(regionEndRadians)
                    //  .padAngle(0.02)
                    .sort(null)
                    .value(1);

                //data = data.sort(function(a,b) { return b.population - a.population});

                data = data.sort(function(a,b){ return d3.ascending(a.cause, b.cause); });
                var g = loc6.selectAll(".arc")
                    .data(pie(data))
                    .enter().append("g")
                    .attr("class", "arc");

                g.append("path")
                    .attr("d", arcPie)
                    .style('fill','none');


                g.append("text")
                    .attr("transform", function(d) {
                        var midAngle = d.endAngle > 0 && d.endAngle < Math.PI ? d.startAngle/2 + d.endAngle/2 : d.startAngle/2  + d.endAngle/2 + Math.PI ;
                        return "translate(" + labelArc.centroid(d)[0] + "," + labelArc.centroid(d)[1] + ") rotate(-90) rotate(" + (midAngle * 180/Math.PI) + ")"; })
                    // .attr("dy", ".35em")
                    .attr('text-anchor',function(d) {
                        return   d.endAngle > 0 && d.endAngle < Math.PI ? 'start' : 'end';
                    })
                    .style('fill', 'black')
                    .text(function(d) { return d.data.cause; });


                function type(d) {
                    d.population = +d.population;
                    return d;
                }

            }
            function addData(data, type, fullArray){
                var finalData = [];
                var dataToUse = d3.nest()
                    .key(function(d){return d.cause;})
                    .entries(data);

                dataToUse.forEach(function(cause){
                    var data = cause.values;
                    var  toAdd = _.difference(fullArray, data.map(function(d){return d.category } ) );

                    data.forEach(function(d){
                        d.isEmpty = false;
                        finalData.push(d);
                    });

                    toAdd.forEach(function(d) {
                        var obj = {
                            'category_type': type,
                            'category': d,
                            'cluster': data[0].cluster,
                            'cause': data[0].cause,
                            'isEmpty': true
                        };
                        finalData.push(obj);
                    })
                })

                return finalData;
            }

            function sortData(data){

                var snapshotData = d3.nest()
                    .key(function(d) { return d.category;})
                    .entries(data)
                    .sort(function(a,b) { return catOrder.indexOf(a) - catOrder.indexOf(b); });

                var b = [];
                snapshotData.forEach(function (d) {
                    b.push(d.values.sort(function (a, b) {
                        return d3.ascending(a.cause, b.cause);
                    }));
                });

                return _.flatten(b.reverse());
            }

            function processCurvedHeatMapData(data){

                var tweetCategoryList = [
                    "promotional",
                    "fundraising",
                    "educational",
                    "personal" ];

                var userCategoryList = [
                    "media",
                    "celebrities",
                    "interest groups",
                    "official agencies",
                    'public',
                    'businesses'
                ];

                var currentData = d3.nest()
                    .key(function(d){ return d.category_type; })
                    .entries(data);

                var user_outData = [], tweet_outData = [];
                currentData.forEach(function(cat_type){
                    if (cat_type.key == 'user_category'){
                        user_outData.push( addData(cat_type.values, 'user_category', userCategoryList) );
                    }
                    else {
                        tweet_outData.push( addData(cat_type.values, 'tweet_category', tweetCategoryList) );
                    }
                });

                user_outData = _.flatten(user_outData);
                tweet_outData = _.flatten(tweet_outData);
                user_outData = sortData(user_outData);
                tweet_outData = sortData(tweet_outData);


                return  [user_outData, tweet_outData];

            }

            function createHeatMapDataForEachCluster(data){

             //   var atHigherLevel = [];
                var clusterData = d3.nest()
                    .key(function(d){ return d.cluster;})
                    .key(function(d){ return d.cause; })
                    .entries(data);

                var output = [];
                clusterData.forEach(function(cluster){
                    cluster.values.forEach(function(cause){
                        output.push(analyzeDataByCategory(cause.values, 'user_category'));
                        output.push(analyzeDataByCategory(cause.values, 'tweet_category'));
                    })
                });

                clusterData.forEach(function(cluster){
                    cluster.values.forEach(function(cause){
                        atHigherLevel.push( analyzeDataOverall(cause.values) );
                    });
                });

                atHigherLevel = _.groupBy(atHigherLevel, function(d){ return d.cluster });
                console.log(atHigherLevel);

                return _.groupBy(_.flatten(output), function(d){ return d.cluster});
            }

            function netSentimentByCategory(data){

                var clusterData = d3.nest()
                    .key(function(d){ return d.cluster;})
                    .entries(data);

                clusterData.forEach(function(el){
                    var data = el.values;
                    var obj  = {};
                    obj.cluster = data[0].cluster;// cluster
                    obj.cause = CAUSE_SUB_GROUP_ABBR_TWITTER[data[0].cluster]; //cause
                    obj.numPos = _.filter(data, function(d) {return d.sentiment == 'positive' }).length;
                    obj.meanPosScore = (obj.numPos == 0) ? 0 : d3.mean(_.filter(data, function(d) {return d.sentiment == 'positive' }), function(d){
                        return d.sentiment_score; });
                    obj.numNeg = _.filter(data, function(d) {return d.sentiment == 'negative' }).length;
                    obj.meanNegScore = (obj.numNeg == 0) ? 0 : d3.mean(_.filter(data, function(d) {return d.sentiment == 'negative' }), function(d){
                        return d.sentiment_score;   });
                    obj.numNeu = _.filter(data, function(d) {return d.sentiment == 'neutral' }).length;
                    obj.numTweets = obj.numPos + obj.numNeg + obj.numNeu;
                    obj.netSentimentValue = (obj.numPos - obj.numNeg)/ obj.numTweets;
                    netPerGroup.push(obj);
                });

                return netPerGroup
            }

            function analyzeDataOverall(data){

                var obj  = {};
                obj.cluster = data[0].cluster;// cluster
                obj.cause = data[0].cause; //cause
                obj.numPos = _.filter(data, function(d) {return d.sentiment == 'positive' }).length;
                obj.meanPosScore = (obj.numPos == 0) ? 0 : d3.mean(_.filter(data, function(d) {return d.sentiment == 'positive' }), function(d){
                    return d.sentiment_score; });
                obj.numNeg = _.filter(data, function(d) {return d.sentiment == 'negative' }).length;
                obj.meanNegScore = (obj.numNeg == 0) ? 0 : d3.mean(_.filter(data, function(d) {return d.sentiment == 'negative' }), function(d){
                    return d.sentiment_score;   });
                obj.numNeu = _.filter(data, function(d) {return d.sentiment == 'neutral' }).length;
                obj.numTweets = obj.numPos + obj.numNeg + obj.numNeu;
                obj.netSentimentValue = (obj.numPos - obj.numNeg)/ obj.numTweets;

                return obj
            }

            function analyzeDataByCategory(data, cat){

                var categoryData = d3.nest()
                    .key(function(d){ return d[cat];})
                    .entries(data);


                var outputData = [];
                categoryData.forEach(function(category){
                    var obj = {};
                    obj.category_type = cat;  //user or tweet
                    obj.category = category.key; //public, media, this or that
                    obj.cluster = category.values[0].cluster;// cluster
                    obj.cause = category.values[0].cause; //cause
                    obj.numPos = _.filter(category.values, function(d) {return d.sentiment == 'positive' }).length;
                    obj.meanPosScore = (obj.numPos == 0) ? 0 : d3.mean(_.filter(category.values, function(d) {return d.sentiment == 'positive' }), function(d){
                        return d.sentiment_score; });
                    obj.numNeg = _.filter(category.values, function(d) {return d.sentiment == 'negative' }).length;
                    obj.meanNegScore = (obj.numNeg == 0) ? 0 : d3.mean(_.filter(category.values, function(d) {return d.sentiment == 'negative' }), function(d){
                        return d.sentiment_score;   });
                    obj.numNeu = _.filter(category.values, function(d) {return d.sentiment == 'neutral' }).length;
                    obj.netSentimentValue = (obj.numPos - obj.numNeg)/ (obj.numPos + obj.numNeg + obj.numNeu);
                    outputData.push(obj);
                });
                return outputData
            }

            drawPie(dataKeywords, svg, outerRadius * 2.05);

        });

    }



    chart.range = function(_) {
        if (!arguments.length) return range;
        range = _;
        return chart;
    };


    return chart;
};




///////////////////// Extra functions //////////////////////

function wrap(text, width) {
  text.each(function() {
	var text = d3.select(this),
		words = text.text().split(/\s+/).reverse(),
		word,
		line = [],
		lineNumber = 0,
		lineHeight = 1.2, // ems
		y = parseFloat(text.attr("y")),
		x = parseFloat(text.attr("x")),
		dy = parseFloat(text.attr("dy")),
		tspan = text.text(null).append("tspan").attr("x", x).attr("y", y).attr("dy", dy + "em");

	while (word = words.pop()) {
	  line.push(word);
	  tspan.text(line.join(" "));
	  if (tspan.node().getComputedTextLength() > width) {
		line.pop();
		tspan.text(line.join(" "));
		line = [word];
		tspan = text.append("tspan").attr("x", x).attr("y", y).attr("dy", ++lineNumber * lineHeight + dy + "em").text(word);
	  }
	}
  });
}//wrap



function jsUcfirst(string) {
    return string.charAt(0).toUpperCase() + string.slice(1);
}


