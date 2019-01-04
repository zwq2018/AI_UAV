define("sjs/util/urlspe", function() {
	/*--
		处理url模块，生成参数对象等
		-as url
		-file 
	*/
	var url = {
		/*--
			name 要查询字符串
			-p string name 要查询参数
			-p string url url地址
			-r string 要查询参数的值
		*/
		query : function(name, u) {
			var reg = new RegExp("(^|&)" + name + "=([^&]*)(&|$)");
			if (u) {
				u = u.substr(u.indexOf('?') + 1);
			} else {
				u = window.location.search.substr(1)
			}
			var r = u.match(reg);
			if (r != null)
				return unescape(r[2]);
			return null;
		},
		/*--
			查询参数对象
			-r obj 参数对象
		*/
		getQueryJson : function() {
			var ret = {}, arr;
			//window.location.search.substr(1).replace(/(\w+)=(\w+)/ig, function(a, b, c){ret[b] = unescape(c);});
			if (!window.location.search)
				return {};

			arr = window.location.search.substr(1).split("&");
			for (var i = 0; i < arr.length; i++) {
				var a = arr[i].split("=") || [];
				ret[a[0]] = a[1];
			}
			return ret;
		},
		/*--
			格式化参数
			-p obj obj 查询参数对象
			-r string reval 构造成查询参数字符串
		*/
		param : function(obj) {

			var reval = '', constr = '';
			for (var pro in obj) {
				reval += (constr + pro + '=' + obj[pro]);
				constr = '&';
			}
			return reval;
		}
	};

	return url;
}); 