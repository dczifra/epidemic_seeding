var graph = {'nodes': [], 'links': []}
var graphDict = {}

var svg = null;

var svg2 = null;
var slider = null;
var sliderHandle = null;
var sliderLabel = null;
var sliderx = null;
var sliderMoving = false;
var sliderTimer = null;

var div = null;
var selectLinks = null;
var selectNodes = null;
var offset = 10;
var scaleX = 65;
var scaleY = 100;
var autoFit = true;
var graphLoaded = false;
var simLoaded = false;

var parameters = null;
var graph_parameters = null;
var init_nodes = null;
var simChart = null;
var simNodeChart = null;
var node_plots= null;
var agg_plots = null;
var rawInputText = null;
var saveData = null;
var reader = null;
var globalTime = -1;
var maxTime = -1;

var nodeToString = function(a) {
    if (graph_parameters["graph_type"]=="grid"){
	    return "("+a[0].toString()+", " +a[1].toString()+")";
	}
	else {
	    return a.toString();
	}
}

$(document).ready(function() {
     // Use a "/test" namespace.
     // An application can open a connection on multiple namespaces, and
     // Socket.IO will multiplex all those connections on a single
     // physical channel. If you don't care about multiple channels, you
     // can set the namespace to an empty string.			
     namespace = '/test';

     // Connect to the Socket.IO server.
     // The connection URL has the following format, relative to the current page:
     //     http[s]://<domain>:<port>[/<namespace>]
     var socket = io(namespace);

     // Event handler for new connections.
     // The callback function is invoked when a connection with the
     // server is established.
     socket.on('connect', function() {
         console.log("Connect")
         socket.emit('my_event', {data: 'I\'m connected!'});
     });

     socket.on('disconnect', function() {
         console.log("Disconnect")
     });


     	zoom = d3.zoom()
		      .scaleExtent([0.1, 4])
		      .on("zoom", zoom);
		      
		width=600
		height=500
         svg= d3.select("#game_board").append("svg")
		    .attr("width",  width)
		    .attr("height",  height)
		    .call(zoom)
		    .attr("z-index",-1)
		    //.call(d3.zoom().on("zoom", function () {
		     //  svg.attr("transform", d3.event.transform)
		    //}));
    
		    
		// Define the div for the tooltip
		div = d3.select("body").append("div")	
		 .attr("class", "tooltip")				
		 .style("opacity", 0);
    
    
        //svg.style("background","url('static/HungaryMap.png') no-repeat");
    
		selectLinks = svg.append("g").attr("class", "links");
		selectNodes = svg.append("g").attr("class", "nodes");
    
    
	d3.select("#fitButton").on("click", function() {
		zoomFit(350);
	 } );
	 
	 
	d3.selectAll(".rulesButton").on("click", function() {
	    $("#rules").modal("show");
 		document.getElementById('doneRulesModal').onclick = function() { 
 			$("#rules").modal("hide");
 		}
 	});


	d3.selectAll(".loadButton").on("click", function() {
	    init_nodes = {};
		document.getElementById("graphParamDiv").style.display = 'none';
        document.getElementById("paramDiv").style.display = '';
        document.getElementById("saveDiv").style.display = 'none';
        document.getElementById("fileDiv").style.display = 'none';

 		graph_parameters =  {
 						"graph_type": document.getElementById("graph_type").value,
  						"graph_args": {"n": parseInt(document.getElementById("n_nodes").value, 10), "N": parseInt(document.getElementById("N_population").value,10), "d": parseInt(document.getElementById("d_degree").value,10)},
     						  };
	
		socket.emit('load_graph',graph_parameters);
		$("#waitbox").modal("show");
     });
     
     d3.selectAll(".loadSimButton").on("click", function() {
		document.getElementById("graphParamDiv").style.display = 'none';
        document.getElementById("paramDiv").style.display = 'none';
        document.getElementById("saveDiv").style.display = 'none';
        document.getElementById("fileDiv").style.display = '';
     });


	d3.selectAll(".unloadButton").on("click", function() {
	    graphLoaded = false;
	    simLoaded = false;
		document.getElementById("graphParamDiv").style.display = '';
        document.getElementById("paramDiv").style.display = 'none';
        document.getElementById("saveDiv").style.display = 'none';
        document.getElementById("fileDiv").style.display = 'none';
        
        graph = {'nodes': [], 'links': []};
        graphDict = {};
        saveData = null;
        node_plots =null;
        agg_lots = null;
        reader= null;
        document.getElementById("file-input").value = "";

        update();
     });
     
     d3.selectAll(".saveButton").on("click", function() {
	    saveUrl = window.URL.createObjectURL(saveData);
	    window.open(saveUrl);
	 });

     d3.select("#playButton").on("click", function() {
        if (simLoaded) {
        	var button = d3.select(this);
	    	if (button.text() == "Pause") {
	    	  sliderMoving = false;
	    	  clearInterval(sliderTimer);
	    	  // timer = 0;
	    	  button.text("Play");
	    	} else {
	    	  sliderMoving = true;
	    	  sliderTimer = setInterval(step, parseInt(document.getElementById("pause_time").value, 10) );
	    	  button.text("Pause");
	    	}
	    	console.log("Slider moving: " + sliderMoving);
	    }
	})

	function readSingleFile(e) {
	  var file = e.target.files[0];
	  console.log(e)
	  if (!file) {
	    return;
	  }
	  reader = new FileReader();
	  reader.onload = function(e) {

	    rawInputText = e.target.result;
	    inputText = rawInputText.split(/\r?\n/);
	    inputData = []
	    for (key in inputText) {
	        if(inputText[key].length>0) {
	        	inputData.push(JSON.parse(inputText[key]))
	        }
	    }
	    saveRawData=inputData;
	    inputDataToTimeSeries(inputData)
	    	    
	   simChart = new Chart({
			element: document.querySelector('#chart-container'),
		    data: agg_plots["d_SIR"],
		    maxLength: 0
		});
        simLoaded = true;
     	loadSlider();  	
     	update();            

	  };
	  reader.readAsText(file);
	}
		
		
	function inputDataToTimeSeries(inputData) {
		node_plots= {}
	    agg_plots = {}
	    Object.keys(inputData[0].node_data).forEach(function (node)  {
	        node_plots[node]={}
	        Object.keys(inputData[0].node_data[node]).forEach(function (record)  {
	           if (record.slice(0,2)=="d_") {
	           	   node_plots[node][record]={}
	               Object.keys(inputData[0].node_data[node][record]).forEach(function (plot) {
	                   node_plots[node][record][plot]=[]
	               });
	           } else if (record.slice(0,2)=="b_") {
	               node_plots[node][record]={};
	           }
	        });
	    });
	    Object.keys(inputData[0].agg_data).forEach(function (record)  {
	        if (record.slice(0,2)=="d_") {
		    	agg_plots[record]={}
		    	Object.keys(inputData[0].agg_data[record]).forEach(function (plot) {
		            agg_plots[record][plot]=[]
		   		});
	        } else if (record.slice(0,2)=="b_") {
	    		 agg_plots[record]={};
	    	}
		});

	    
	    for (key in inputData) {
		    Object.keys(inputData[key].node_data).forEach(function (node)  {
		        Object.keys(inputData[key].node_data[node]).forEach(function (record)  {
		           if (record.slice(0,2)=="d_") {
		               Object.keys(inputData[key].node_data[node][record]).forEach(function (plot) {
		                   node_plots[node][record][plot].push(inputData[key].node_data[node][record][plot])
		               });
		           } else if (record.slice(0,2)=="b_") {
		               node_plots[node][record][key]=inputData[key].node_data[node][record];
		           }
		        });
		    });
		    
		    Object.keys(inputData[key].agg_data).forEach(function (record)  {
			    if (record.slice(0,2)=="d_") {
		    		Object.keys(inputData[key].agg_data[record]).forEach(function (plot) {
		                agg_plots[record][plot].push(inputData[key].agg_data[record][plot])
		        	});
		        } else if (record.slice(0,2)=="b_") {
		        	agg_plots[record][key]=inputData[key].agg_data[record]
		        }
		        
		    });
        }
        maxTime = inputData.length;
	}	
		
		
   document.getElementById('file-input')
     .addEventListener('change', readSingleFile, false);

	d3.selectAll(".simButton").on("click", function() {
	    document.getElementById("saveDiv").style.display = '';

 		parameters =  getParams();

		socket.emit('simulate',parameters);
		
		$("#waitbox").modal("show");
     });


	d3.selectAll(".compAwButton").on("click", function() {
 		parameters =  getParams();
		socket.emit('compAw',{"data": saveRawData, "parameters":parameters});
		$("#waitbox").modal("show");
     });


	 
/*	d3.select("#zoomInButton").on("click", function() {zoomScale(350,1.2) } );
	d3.select("#zoomOutButton").on("click", function() {zoomScale(350,0.8) } );*/


     // Event handler for server sent data.
     // The callback function is invoked whenever the server emits data
     // to the client. The data is then displayed in the "Received"
     // section of the page.
     socket.on('my_response', function(msg, cb) {
      //   $('#log').append('<br>' + $('<div/>').text('Received #' + msg.count + ': ' + msg.data).html());
         if (cb)
             cb();
     });
     
     socket.on('waitbox_response', function(msg, cb) {
     	d3.select("#waitperc").html(msg);
         if (cb)
             cb();
     });


     socket.on('graph_response', function(msg, cb) {
        $("#waitbox").modal("hide");
	    graphLoaded = true;
		graph['links'] = graph['links'].concat(msg['graph']['links']);
		msg['graph']['nodes'].forEach(function (item, index) {
		        currentID=item['id']
		        currentState="S"
		        graph['nodes'].push({state: currentState, init_num: 0 , label: item['label']  , index: item['index']  , pos: item['pos']  ,id: currentID})
 	    		graphDict[currentID] = {state: currentState, init_num: 0 , label: item['label']  , index: item['index'], pos: item['pos']  , id: currentID, neighs : {}};
 	    });
     	update();            
		zoomFit(350,0)

		
		if (cb)
             cb();

     });


     socket.on('sim_response', function(msg, cb) {
     	d3.select("#waitperc").html("0%");
        $("#waitbox").modal("hide");
	    simLoaded = true;
        console.log(msg)
        saveRawData=msg["result"];
        inputDataToTimeSeries(msg["result"]);
        saveText = ""
        for (key in msg["result"]) {
        	saveText+=JSON.stringify(msg["result"][key]) + "\n";
        }
    	saveData = new Blob([saveText], {type: 'text/plain'});

                
        console.log(msg)
		simChart = new Chart({
			element: document.querySelector('#chart-container'),
		    data: agg_plots["d_SIR"],
		    maxLength: 0
		});
		
		graph['links'] = graph['links'].concat(msg['graph']['links']);
		msg['graph']['nodes'].forEach(function (item) {
		        currentID=item['id']
		        graph['nodes'].push({label: item['label'] , index: item['index']  , init_num: item['init_node'],   pos: item['pos']  ,id: currentID})
 	    		graphDict[currentID] = {label: item['label'], index: item['index']  , init_num: item['init_node'], pos: item['pos'] , id: currentID, neighs : {}};
 	    });
     	 
     	loadSlider();  	
     	update();            
		zoomFit(350,0)
		
		if (cb)
             cb();

     });
     
          
     
     function getParams() {
          return {      "model_type": document.getElementById("model_type").value,
 						"graph_type": document.getElementById("graph_type").value,
  						"graph_args": {"n": parseInt(document.getElementById("n_nodes").value, 10), "N": parseInt(document.getElementById("N_population").value,10), "d": parseInt(document.getElementById("d_degree").value,10)},
					    "logfile": "example1.csv",
					    "max_iteration": parseInt(document.getElementById("num_iterations").value, 10),
					    "init_strategy": "init_nodes",
					    "init_nodes": init_nodes,
					    "s0": parseInt(document.getElementById("s0").value, 10),
					    "beta": parseFloat(document.getElementById("beta").value),
					    "beta_super": parseFloat(document.getElementById("beta_super").value),
					    "sigma": parseFloat(document.getElementById("sigma").value),
					    "gamma": parseFloat(document.getElementById("gamma").value),
					    "xi": parseFloat(document.getElementById("xi").value),
					    "MOVING_WILLINGNESS": parseFloat(document.getElementById("MOVING_WILLINGNESS").value),
					    "MAX_E_TIME":10,
					    "MAX_I_TIME":10,
					    "p_worker": parseFloat(document.getElementById("p_worker").value),
					    "p_teleport":parseFloat(document.getElementById("p_teleport").value),
					    "p_super":parseFloat(document.getElementById("p_super").value),
					    "awM": parseFloat(document.getElementById("aw_m").value),
					    "awR": parseFloat(document.getElementById("aw_r").value)
     						  };
     }


     function update() {			
         var links = selectLinks.selectAll(".link")
		  .data(graph['links'])
        
		links.enter().append("line")
		   .merge(links)
		      .attr("x1", d=> graphDict[d["source"]]['pos'][0]*scaleX+offset)
		      .attr("x2", d=> graphDict[d["target"]]['pos'][0]*scaleX+offset)
		      .attr("y1", d=> -graphDict[d["source"]]['pos'][1]*scaleY+offset)
		      .attr("y2", d=> -graphDict[d["target"]]['pos'][1]*scaleY+offset)
			  .attr("class", "link")
		      .style("stroke", "steelblue")
		      .style("stroke-width", 4)

		
		var nodes =selectNodes.selectAll(".node")
		  .data(graph['nodes'], d=> d.id);
		  
		nodes.enter().append("circle")
		   .merge(nodes)
		    .attr("cx", d => d['pos'][0]*scaleX+offset)
		    .attr("cy", d => -d['pos'][1]*scaleY+offset)
		    .attr("r", 20)
			.on('mouseover', mouseOverFunction)
		    .on('mouseout', mouseOutFunction)
		    .attr("class", "node")
		    .style("fill", function(d) {
		        if (graphDict[d['id']]['init_num']>0) {
		            return "purple"
		        }
		        else if (simLoaded) {
		            if (globalTime<0) {
		    			if(Math.max.apply(false, node_plots[d["index"]]["d_SIR"]["I"]) >0) { return "red";}
		    			else if (("G" in node_plots[d["index"]]["d_SIR"]) && (Math.max.apply(false, node_plots[d["index"]]["d_SIR"]["G"]) >0)) { return "black"; }
		    			else {return d3.rgb(229, 248, 255);}
		    		}
		    		else {
		    			if(node_plots[d["index"]]["d_SIR"]["I"][globalTime]>0) { return "red";}
		    			else if (("G" in node_plots[d["index"]]["d_SIR"]) && (node_plots[d["index"]]["d_SIR"]["G"][globalTime]>0)) { return "black"; }
		    			//else if (("R" in node_plots[d["index"]]["d_SIR"]) && (node_plots[d["index"]]["d_SIR"]["R"][globalTime]>0)) { return "blue"; }
		    			else {return d3.rgb(229, 248, 255);}
		    		}
		    	}
		    	else {
		    	    return d3.rgb(229, 248, 255)
		    	}
		    })
		    .style("stroke", "black")
		    .on("click", function(d) {
		            if (!simLoaded) {
		                graphDict[d['id']]['init_num']+=parseInt(document.getElementById("s0").value, 10);
		                init_nodes[graphDict[d['id']]['index']]= graphDict[d['id']]['init_num'];
		                d3.select("#node-chart-container").html("Init inf: "+graphDict[d['id']]['init_num'].toString())
		                update();
		    		}
		    });
		
		links.exit().remove();
		nodes.exit().remove();
		
		////////// Awareness plots
		if (simLoaded && ("d_aw" in agg_plots)) {
     			if (globalTime<0) {
			       simNodeChart = new Chart({
					  element: document.querySelector('#aw-chart-container'),
				      data: agg_plots["d_aw"],
				      maxLength: 0
				   });
			     }
			     else {
			   	   simNodeChart = new BarPlot({
					  element: document.querySelector('#aw-chart-container'),
				      data: agg_plots["b_aw"][globalTime],
				      maxLength: 1
				   });
				 }
		} 
	};
	
				
	function zoomFit(transitionDuration) {
        //scaleX=parseFloat(document.getElementById("xScale").value)
        //update()
	    var bounds = selectNodes.node().getBBox();
	    var fullWidth = svg.node().width["baseVal"].value,
	        fullHeight = svg.node().height["baseVal"].value;
	    var width = bounds.width,
	        height = bounds.height;
	    var midX = bounds.x + width / 2,
	        midY = bounds.y + height / 2;
	    var scale = 0.6 / Math.max(width / fullWidth, height / fullHeight);
	    var translate = [fullWidth / 2 - scale * midX, fullHeight / 2 - scale * midY];
	    if (width == 0 || height == 0) return;
	    if (selectNodes.attr("transform")!=null) {
		  var oldTranslate = [parseFloat(selectNodes.attr("transform").split("(")[1].split(",")[0]),parseFloat(selectNodes.attr("transform").split("(")[1].split(",")[1])]
	      var oldScale = parseFloat(selectNodes.attr("transform").split("(")[2])
	      console.log(scale, oldScale, translate[0],oldTranslate[0] , translate[1],oldTranslate[1])
	      if (scale> oldScale) return; // nothing to fit
	    }
	    					    
	    svg
	        .transition()
	        .duration(transitionDuration || 0) // milliseconds
	        .call(zoom.transform, d3.zoomIdentity.translate(translate[0],translate[1]).scale(scale));
	}


/*	function zoomScale(transitionDuration,multiplier) {
		var oldTranslate = [parseFloat(selectNodes.attr("transform").split("(")[1].split(",")[0]),parseFloat(selectNodes.attr("transform").split("(")[1].split(",")[1])]
	    var oldScale = parseFloat(selectNodes.attr("transform").split("(")[2])
	    console.log(oldTranslate,oldScale)
		svg
	        .transition()
	        .duration(transitionDuration || 0) // milliseconds
	        .call(zoom.transform, d3.zoomIdentity.scale(multiplier));
	}*/
	
	function zoom() {
	  selectLinks.attr("transform", (d3.event.transform));
	  selectNodes.attr("transform", (d3.event.transform));
	}



	const mouseOverFunction = function (d) {
	  const circle = d3.select(this);
	
	    
	  toolText="Node: (" + d['label'] + ")<br>";
	  toolText+="<div id='node-chart-container' style='width: 500px;' display='block'></div>"
	  
	  div.transition()		
	         .duration(200)		
	         .style("opacity", 1);		
	   div.html(toolText)	
	         .style("left", (d3.event.pageX + 10) + "px")		
	         .style("top", (d3.event.pageY+ 10) + "px");
	    
	  if (simLoaded) {
	       if (document.getElementById("node_plot_type").value in node_plots[d["index"]]) {
	           if (document.getElementById("node_plot_type").value.slice(0,2)=="d_") {
	        	 simNodeChart = new Chart({
					element: document.querySelector('#node-chart-container'),
				    data: node_plots[d["index"]][document.getElementById("node_plot_type").value],
				    maxLength: 0
				 });
			   }
			   else if (document.getElementById("node_plot_type").value.slice(0,2)=="b_") {
			     if (globalTime<0) {
			       simNodeChart = new Chart({
					  element: document.querySelector('#node-chart-container'),
				      data: node_plots[d["index"]]["d_"+document.getElementById("node_plot_type").value.slice(2)],
				      maxLength: 0
				   });
			     }
			     else {
			       console.log(node_plots[d["index"]][document.getElementById("node_plot_type").value][globalTime])
			   	   simNodeChart = new BarPlot({
					  element: document.querySelector('#node-chart-container'),
				      data: node_plots[d["index"]][document.getElementById("node_plot_type").value][globalTime],
				      maxLength: 0
				   });
				 }
			   }	
			}
	  } 
	  else {
	       d3.select("#node-chart-container").html("Init inf: "+graphDict[d['id']]['init_num'].toString())
	  }
	
	};
	
	var idEq = function(a,b) {	return ((a[0]===b[0]) & (a[1]===b[1])) }
	const mouseOutFunction = function () {   
	  div.transition()		
	         .duration(500)		
	         .style("opacity", 0);	
	
	};
	
	function loadSlider() { 
			console.log("Slider") 
			globalTime=-1;
			d3.select("#slider").select("svg").remove();
			svg2 = d3.select("#slider")
			    .append("svg")
			    .attr("width", 500)
			    .attr("height", 50);
					    		
			sliderx = d3.scaleLinear()
				.domain([0, maxTime])
				.range([0, 450])
				.clamp(true);
		
			slider = svg2.append("g")
			    .attr("class", "slider")
			    .attr("transform", "translate(" + 10 + "," + 40 + ")");
			
			slider.append("line")
			    .attr("class", "track")
			    .attr("x1", sliderx.range()[0])
			    .attr("x2", sliderx.range()[1])
			  .select(function() { return this.parentNode.appendChild(this.cloneNode(true)); })
			    .attr("class", "track-inset")
			  .select(function() { return this.parentNode.appendChild(this.cloneNode(true)); })
			    .attr("class", "track-overlay")
			    .call(d3.drag()
			        .on("start.interrupt", function() { slider.interrupt(); })
			        .on("start drag", function() {
			          currentValue = d3.event.x;
			          updateSlider(sliderx.invert(currentValue)); 
			        })
			    );
			
			slider.insert("g", ".track-overlay")
			    .attr("class", "ticks")
			    .attr("transform", "translate(0," + 18 + ")")
			  .selectAll("text")
			    .data(sliderx.ticks(10))
			    .enter()
			    .append("text")
			    .attr("x", sliderx)
			    .attr("y", 10)
			    .attr("text-anchor", "middle")
	
			sliderHandle = slider.insert("circle", ".track-overlay")
			    .attr("class", "handle")
			    .attr("r", 9);
			
			sliderLabel = slider.append("text")  
			    .attr("class", "label")
			    .attr("text-anchor", "left")
			    .text("Red signifies nodes that at least once had an infected agent")
			    .attr("transform", "translate(0," + (-25) + ")")
	}

	function updateSlider(h) {
	  // update position and text of label according to slider scale
	  sliderHandle.attr("cx", sliderx(h));
	  globalTime = Math.round(h)
	  sliderLabel
	    .attr("x", sliderx(h))
	    .text("Time: " + globalTime.toString());
	   update();
	}
	
	function step() {
	  if (globalTime+1.001 > maxTime) {
	    sliderMoving = false;
	    globalTime = -1;
	    clearInterval(sliderTimer);
	    // timer = 0;
	    d3.select("#playButton").text("Play");
	    console.log("Slider moving: " + sliderMoving);
	    sliderHandle.attr("cx", sliderx(0));
	    sliderLabel
	    	.attr("x", sliderx(0))
	    	.text("Red signifies nodes that at least once had an infected agent");
	    update();
	  }
	  else {
	  	  updateSlider(globalTime+1);
	  }
	}



     // Interval function that tests message latency by sending a "ping"
     // message. The server then responds with a "pong" message and the
     // round trip time is measured.
     var ping_pong_times = [];
     var start_time;
     window.setInterval(function() {
         start_time = (new Date).getTime();
         socket.emit('my_ping');
     }, 1000);

     // Handler for the "pong" message. When the pong is received, the
     // time from the ping is stored, and the average of the last 30
     // samples is average and displayed.
     socket.on('my_pong', function() {
         var latency = (new Date).getTime() - start_time;
         ping_pong_times.push(latency);
         ping_pong_times = ping_pong_times.slice(-30); // keep last 30 samples
         var sum = 0;
         for (var i = 0; i < ping_pong_times.length; i++)
             sum += ping_pong_times[i];
        console.log("pong")
        // $('#ping-pong').text(Math.round(10 * sum / ping_pong_times.length) / 10);
     });

     // Handlers for the different forms in the page.
     // These accept data from the user and send it to the server in a
     // variety of ways
     $('form#emit').submit(function(event) {
         socket.emit('my_event', {data: $('#emit_data').val()});
         return false;
     });
         

})