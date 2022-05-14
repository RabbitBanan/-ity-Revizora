var latitude = [44.937157, 44.936293, 44.932396, 44.923344]
var longitude = [37.311268, 37.311926, 37.316914, 37.320245]
var issue_class = ['Загрязненный дорожный знак', 'Препятствие на дороге', 'Стерта дорожная разметка', 'Нарушение целостности дорожной неровности']
var issue_img = ['img1.png', 'img2.png', 'img3.png', 'img4.png']
var issue_status = ['issue-repeat', 'issue-new', 'issue-new', 'issue-user']
var issue_id = ['issue-1', 'issue-2', 'issue-3', 'issue-4']
var issue_dates = ['12.04.2022', '13.05.2022', '13.05.2022', '10.05.2022']
var issue_stat_color = new Map([['issue-repeat', '#f98080'], ['issue-new', '#f9cd80'], ['issue-user', '#d0bdf0']]);
var issue_stat_type = new Map([['issue-repeat', 'Старый'], ['issue-new', 'Новый'], ['issue-user', 'От гражданина']]);
//
function calculateRouteFromAtoB(platform) {
    for (var i = 1; i < latitude.length; i++) {
        var router = platform.getRoutingService(null, 8),
            routeRequestParams = {
                routingMode: 'fast',
                transportMode: 'car',
                origin: latitude[i - 1] + ',' + longitude[i - 1],
                destination: latitude[i] + ',' + longitude[i],
                return: 'polyline,turnByTurnActions,actions,instructions,travelSummary'
            };

        router.calculateRoute(
            routeRequestParams,
            onSuccess,
            onError
        );
    }
}

function onSuccess(result) {
    var route = result.routes[0];
        addRouteShapeToMap(route);
}

function onError(error) {
    alert('Can\'t reach the remote server');
}

function addRouteShapeToMap(route) {
    route.sections.forEach((section) => {
        let linestring = H.geo.LineString.fromFlexiblePolyline(section.polyline);

        let polyline = new H.map.Polyline(linestring, {
            style: {
                lineWidth: 8,
                strokeColor: 'rgba(0, 128, 255, 0.7)'
            }
        });

        map.addObject(polyline);
        map.getViewModel().setLookAtData({
            bounds: polyline.getBoundingBox()
        });
    });
}
//
function read_select_value() {
    var select = document.getElementById('issue-select-filter');
    filter_issue_container(select.options[select.selectedIndex].value);
}

function issue_click_handler(evt) {
    parent_elem_id = Number(evt.target.id.slice(6)) - 1;

    container = document.getElementById('issue-container');
    while (container.firstChild) {
        container.removeChild(container.firstChild);
    }

    issue_detail_container = document.createElement('div');
    issue_detail_container.className = 'issue-detail';
    issue_detail_container.id = evt.target.id;
    container.appendChild(issue_detail_container);

    exit_button = document.createElement('button');
    exit_button.className = 'issue-detail-exit-button';
    exit_button.innerHTML = '<svg version="1.1" viewBox="0 0 12 12">' +
                            '<path d="m8.12 6 3.66-3.66c.29-.29.29-.76 0-1.05l-1.06-1.06c-' +
                            '.29-.29-.76-.29-1.05 0l-3.66 3.66-3.66-3.66c-.29-.29-.76-.29-1.05' +
                            ' 0l-1.06 1.06c-.29.29-.3.76 0 1.05l3.66 3.66-3.66 3.66c-.29.29-.29.76' +
                            ' 0 1.05l1.06 1.06c.29.29.76.29 1.05 0l3.66-3.66 3.66 3.66c.29.29.76.29' +
                            ' 1.05 0l1.06-1.06c.29-.29.3-.76 0-1.05z"></path></svg>';
    exit_button.addEventListener('click', read_select_value);
    issue_detail_container.appendChild(exit_button);

    issue_detail_title = document.createElement('div');
    issue_detail_title.className = 'issue-detail-title ' + issue_status[parent_elem_id];
    issue_detail_title.innerHTML = '<p>' + issue_class[parent_elem_id] + '</p>';
    issue_detail_container.appendChild(issue_detail_title);

    issue_detail_picture = document.createElement('div');
    issue_detail_picture.className = 'issue-detail-picture';
    issue_detail_picture.innerHTML = '<img src="static/res/' + issue_img[parent_elem_id] + '">';
    issue_detail_container.appendChild(issue_detail_picture);

    issue_detail_coordinates = document.createElement('div');
    issue_detail_coordinates.className = 'issue-detail-coordinates';
    issue_detail_coordinates.innerHTML = 'Координаты: ' + latitude[parent_elem_id] + '°, ' + longitude[parent_elem_id] + '°';
    issue_detail_container.appendChild(issue_detail_coordinates);

    issue_detail_status = document.createElement('div');
    issue_detail_status.className = 'issue-detail-status';
    issue_detail_status.innerHTML = 'Статус: ' + issue_stat_type.get(issue_status[parent_elem_id]);
    console.log(issue_stat_type.get(issue_status[parent_elem_id]));
    issue_detail_container.appendChild(issue_detail_status);

    issue_detail_date = document.createElement('div');
    issue_detail_date.className = 'issue-detail-date';
    issue_detail_date.innerHTML = 'Дата обнаружения: ' + issue_dates[parent_elem_id];
    issue_detail_container.appendChild(issue_detail_date);
}

function addIssuesToControl() {
    list = document.getElementById('issue-container');
    for (var i = 0; i < latitude.length; i++) {
        var child = document.createElement('div');
        child.id = issue_id[i];
        child.className = 'issue-element ' + issue_status[i];
        child.innerHTML = issue_class[i] + ' по координатам:<br>' + latitude[i] + '°, ' + longitude[i] + '°';
        child.addEventListener('click', issue_click_handler);
        list.appendChild(child);
    }
}

function addMarkerToGroup(group, coordinate, status, html) {
    var svgMarkup = '<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="42px" height="42px" viewBox="-4 0 36 36" version="1.1">' +
                    '<path d="M14,0 C21.732,0 28,5.641 28,12.6 C28,23.963 14,36 14,36 C14,36 0,24.064 0,12.6 C0,5.641 6.268,0 14,0 Z" id="Shape" fill="${FILL}"/>' +
                    '<circle id="Oval" fill="#01b6b2" fill-rule="nonzero" cx="14" cy="14" r="5"/>' +
                    '</svg>';
    var marker_color = issue_stat_color.get(status);
    svgMarkup = svgMarkup.replace('${FILL}', marker_color);

    var my_icon = new H.map.Icon(svgMarkup);
    var marker = new H.map.Marker(coordinate, {icon: my_icon});

    marker.setData(html);
    group.addObject(marker);
}

function addInfoBubble(map) {
    var group = new H.map.Group();
    map.addObject(group);
    group.addEventListener('tap', function (evt) {
        var bubble = new H.ui.InfoBubble(evt.target.getGeometry(), {
            content: evt.target.getData()
        });
        ui.addBubble(bubble);
    }, false);

    for (var i = 0; i < latitude.length; i++) {
        addMarkerToGroup(group, {lat:latitude[i], lng:longitude[i]}, issue_status[i],
            '<div id="bubble-' + issue_id[i] + '" ' +
            'class="issue-info ' + issue_status[i] + ' issue-bubble-header">' + issue_class[i] + ' по координатам:<br>' +
            latitude[i] + '°, ' + longitude[i] + '°' + '</div>' +
            '<div class="issue-img-container"><img class="issue-photo" src="static/res/' + issue_img[i] + '"></img></div>'
        );
    }
}

var platform = new H.service.Platform({
    'apikey': '_SORhvcov-x7-QsXLs-PvFoGD013ZzzfWX2MgaViYZk'
});

var defaultLayers = platform.createDefaultLayers({lg: 'ru'});

var map = new H.Map(document.getElementById('map'), defaultLayers.vector.normal.map, {
    center: new H.geo.Point(latitude[0], longitude[0]),
    zoom: 16,
    pixelRatio: window.devicePixelRatio || 1
});

window.addEventListener('resize', () => map.getViewPort().resize());

var behavior = new H.mapevents.Behavior(new H.mapevents.MapEvents(map));
var ui = H.ui.UI.createDefault(map, defaultLayers);

addInfoBubble(map);
addIssuesToControl();
calculateRouteFromAtoB(platform);

function filter_issue_container(selected_value) {
    list = document.getElementById('issue-container');
    if (list) {
        while (list.firstChild) {
            list.removeChild(list.firstChild);
        }
        if (!selected_value.length) {
            for (var i = 0; i < latitude.length; i++) {
                var child = document.createElement('div');
                child.id = issue_id[i];
                child.className = 'issue-element ' + issue_status[i];
                child.innerHTML = issue_class[i] + ' по координатам:<br>' + latitude[i] + '°, ' + longitude[i] + '°';
                child.addEventListener('click', issue_click_handler);
                list.appendChild(child);
            }
        } else {
            for (var i = 0; i < latitude.length; i++) {
                if (selected_value == issue_class[i]) {
                    var child = document.createElement('div');
                    child.id = issue_id[i];
                    child.className = 'issue-element ' + issue_status[i];
                    child.innerHTML = issue_class[i] + ' по координатам:<br>' + latitude[i] + '°, ' + longitude[i] + '°';
                    child.addEventListener('click', issue_click_handler);
                    list.appendChild(child);
                }
            }
        }
    }
}
