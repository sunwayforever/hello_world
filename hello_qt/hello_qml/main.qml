import QtQuick 2.14
import QtQuick.Window 2.14
import QtQuick.Controls 2.14
import QtQuick.Layouts 1.0
import sunway.message_box 1.0

Window {
    visible: true
    width: 640
    height: 480
    title: qsTr("Hello World")

    signal windowChanged()
    MessageBox {
        id: message
        height: 1
        width: 1
        onHeightChanged: console.log("height changed "+height)
    }

    Connections {
        target: message
        onChangedTwoTimes: console.log("height changed 2 times")
    }

    Connections {
        target: message
        onOopsWidthChanged: console.log("oops width changed")
    }

    Connections {
        target: a
        onTextChanged: update_with_js()
    }

    Connections {
        target: a
        onTextChanged: windowChanged()
    }

    TextEdit {
        id: a
        x: 70
        y: 95
        width: 80
        height: 20
        text: qsTr("a")
        wrapMode: Text.NoWrap
        font.pixelSize: 12
    }

    TextEdit {
        id: b
        x: 192
        y: 95
        width: 80
        height: 20
        text: qsTr("b")
        wrapMode: Text.NoWrap
        font.pixelSize: 12
    }

    TextEdit {
        id: c
        x: 311
        y: 95

        width: 80
        height: 20
        text: qsTr("c")
        wrapMode: Text.NoWrap
        font.pixelSize: 12

    }

    Connections {
        target: c
        onTextChanged: backend.onTextChanged(c.text)
    }

    Connections {
        target: backend
        onInterrupted: print("backend interrupted")
    }

    Connections {
        target: c
        onTextChanged: windowChanged()
    }

    TextEdit {
        id: d
        x: 411
        y: 95

        width: 80
        height: 20
        text: "d"
        wrapMode: Text.NoWrap
        font.pixelSize: 12

        onTextChanged: message.onTextChanged(d.text)
    }

    Binding {
        target: d
        property: "text"
        value: {
            if (c.text.length > 2) {return c.text}  else {return c.text+c.text}
        }
    }

    Connections {
        target: d
        onTextChanged: windowChanged()
    }

    Button {
        id:button
        onClicked: {
            print("button clicked")
            d.text=Qt.binding(function() {return c.text+c.text+c.text})
        }
    }

    function update_with_js () {
        b.text=a.text
        message.height += 1
        message.width += 1
        if (message.height == 2) {
            message.changedTwoTimes()
            message.foo()
        }
    }
}
