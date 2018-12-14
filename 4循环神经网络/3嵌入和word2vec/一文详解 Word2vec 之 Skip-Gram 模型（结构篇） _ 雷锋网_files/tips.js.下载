(function(w,d,b){

	var YP_WINS = {
		cssUrl : BASE_URL+"/resCommon/css/yp_tipBoxes/tips.css"
	}
	
	/*插入样式文件*/ 
	YP_WINS.fn_addClass = function(){
		var link = d.createElement("link");
		link.rel = "stylesheet";
		link.href = this.cssUrl+'?cdnversion='+~(-new Date()/36e5);
		var head = d.head || d.getElementsByTagName("head")[0] || d.documentElement;
		head.appendChild(link);
	}
	/*三类提示框*/ 
	YP_WINS.fn_tipAction = function(msg,className){
		// $("<div>").addClass("yp_wins_overlay").appendTo("body");
		$("<div>").addClass("yp_win yp_wins_tips"+" "+className).html("<i></i><span>"+msg+"</span>").appendTo("body");
        var l=($(w).width()-$(".yp_wins_tips").outerWidth())/2;
        var t=($(w).height()-$(".yp_wins_tips").outerHeight())/2;
        $(".yp_wins_tips").css({left:l,top:t}).addClass("ani_show");
        setTimeout(function(){
            $(".yp_wins_tips").removeClass("ani_show").addClass("ani_hide");
            // $(".yp_wins_overlay").hide()
        },2000)
        setTimeout(function(){
            $(".yp_wins_tips").remove();
            // $(".yp_wins_overlay").remove();
        },3000)
	}


    /*alert框*/ 
    YP_WINS.fn_alert = function(msg){
		$("<div>").addClass("yp_wins_overlay").appendTo("body");
		$("<div>").addClass("yp_win yp_wins_alert").html("<p>"+msg+"</p><div><a class='close' href='javascript:;'>确认</a></div>").appendTo("body");
        var l=($(w).width()-$(".yp_wins_alert").outerWidth())/2;
        var t=($(w).height()-$(".yp_wins_alert").outerHeight())/2;
        $(".yp_wins_alert").css({left:l,top:t}).addClass("ani_show");
        $(".yp_wins_alert .close").on('click',function(){
            $(".yp_wins_alert").removeClass("ani_show").addClass("ani_hide");
            $(".yp_wins_overlay").hide();
	        setTimeout(function(){
	            $(".yp_wins_alert").remove();
	            $(".yp_wins_overlay").remove()
	        },500)
		})
    }
   
    /*confirm框*/ 
	YP_WINS.fn_confirm = function(obj){
		$("<div>").addClass("yp_wins_overlay").appendTo("body");
		$("<div>").addClass("yp_win yp_wins_confirm").html("<p>"+obj.content+"</p><div class='btns'><a class='true' data-ok='true' href='javascript:;'>确认</a><a class='false' data-ok='false' href='javascript:;'>取消</a><div>").appendTo("body");
        var l=($(w).width()-$(".yp_wins_confirm").outerWidth())/2;
        var t=($(w).height()-$(".yp_wins_confirm").outerHeight())/2;
        $(".yp_wins_confirm").css({left:l,top:t}).addClass("ani_show");
        $(".yp_wins_confirm .btns a").on('click',function(){
        	if($(this).data("ok")){
        		obj.yes();
        	}else{
        		obj.no();
        	}
            $(".yp_wins_confirm").removeClass("ani_show").addClass("ani_hide");
            $(".yp_wins_overlay").hide();
	        setTimeout(function(){
	            $(".yp_wins_confirm").remove();
	            $(".yp_wins_overlay").remove()
	        },500)
		})

    }

    /*confirm框*/ 
	YP_WINS.fn_prompt = function(obj){
		$("<div>").addClass("yp_wins_overlay").appendTo("body");
		$("<div>").addClass("yp_win yp_wins_prompt").html("<p>"+obj.content+"</p><input type='text' value="+obj.default+"><div ><a class='close' href='javascript:;'>确认</a><div>").appendTo("body");
        var l=($(window).width()-$(".yp_wins_prompt").outerWidth())/2;
        var t=($(window).height()-$(".yp_wins_prompt").outerHeight())/2;
        $(".yp_wins_prompt").css({left:l,top:t}).addClass("ani_show");
        $(".yp_wins_prompt .close").on('click',function(){
        	obj.ok($(".yp_wins_prompt input").val());
            $(".yp_wins_prompt").removeClass("ani_show").addClass("ani_hide");
            $(".yp_wins_overlay").hide();
	        setTimeout(function(){
	            $(".yp_wins_prompt").remove();
	            $(".yp_wins_overlay").remove()
	        },500)
		})

    }

    /*初始化*/
	var init = function(){
		// 加载样式
		YP_WINS.fn_addClass(YP_WINS.cssUrl);
	} 
	init();


	/*正确提示*/
	w.YP_tip = function(msg){
		YP_WINS.fn_tipAction(msg,'tip');
	}
	/*警告提示*/
	w.YP_warn = function(msg){
		YP_WINS.fn_tipAction(msg,'warn');
	}
	/*错误提示*/
	w.YP_error = function(msg){
		YP_WINS.fn_tipAction(msg,'error');
	}
	/*alert*/
	w.YP_alert = function(msg){
		YP_WINS.fn_alert(msg);
	}
	/*confirm*/
	w.YP_confirm = function(obj){
		YP_WINS.fn_confirm(obj);
	}
	/*confirm*/
	w.YP_prompt = function(obj){
		YP_WINS.fn_prompt(obj);
	}

})(window,document,document.getElementsByTagName("body")[0]);
