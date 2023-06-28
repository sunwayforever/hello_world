import QtQuick
import QtQuick3D

Window {
    width: 640
    height: 480
    visible: true
    title: qsTr("Hello World")
    View3D {
        id: view
        anchors.fill: parent

        environment: SceneEnvironment {
            clearColor: "skyblue"
            backgroundMode: SceneEnvironment.Color
        }

        Model {
            position: Qt.vector3d(0, 0, 0)
            source: "#Sphere"
            scale: Qt.vector3d(1, 1, 1)
            materials: [ DefaultMaterial {diffuseColor: "green"}]
            SequentialAnimation on x {
                loops: 2
                NumberAnimation {
                    duration: 5000
                    to: -150
                    from: 150
                }
                NumberAnimation {
                    duration: 5000
                    to: 150
                    from: -150
                }
            }
        }

        PerspectiveCamera {
            position: Qt.vector3d(0, 200, 300)
            eulerRotation.x: -30
        }

        DirectionalLight {
            eulerRotation.x: -30
            eulerRotation.y: -70
        }


    }
}
