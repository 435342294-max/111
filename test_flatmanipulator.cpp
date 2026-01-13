#include <osgDB/ReadFile>
#include <osgGA/KeySwitchMatrixManipulator>
#include <osgGA/TrackballManipulator>

#include "utils/View_helper.h"
#include "utils/FlatManipulator.h"

/*!
 * \brief Camera node keyboard event callback handler.
 */
class Projection_callback : public osg::NodeCallback {
public:
  /*!
   * \brief Constructor.
   */
  Projection_callback() {}

  /*!
   * \brief Toggle node projection.
   * \param node callback node
   * \param nv node visitor
   */
  virtual void operator()(osg::Node *node, osg::NodeVisitor *nv) {
    // camera node
    auto camera = static_cast<osg::Camera *>(node);
    // convert the node visitor to an osg::EventVisitor pointer
    const auto ev = nv->asEventVisitor();
    if (ev) {
      // handle events with the node
      for (const auto e : ev->getEvents()) {
        const auto ea = e->asGUIEventAdapter();
        // number key 5, similar to meshlab
        if (ea->getEventType() == osgGA::GUIEventAdapter::KEYDOWN
          && ea->getKey() == '5') {
          auto view = static_cast<osgViewer::View *>(camera->getView());
          if (view) {
            // switch projection mode
            m_orthographic = !m_orthographic;
            // TODO: right and top position w.r.t current view
            // not always scene bounding radius, like meshlab
            auto radius = view->getSceneData()->asGroup()->getBound().radius();
            auto aspect_ratio = camera->getViewport()->aspectRatio();
            if (m_orthographic) {
              // set projection as orthographic
              const auto right = aspect_ratio < 1.0 ? radius : radius * aspect_ratio;
              const auto top = aspect_ratio > 1.0 ? radius : radius / aspect_ratio;
              camera->setProjectionMatrixAsOrtho(-right, right, -top, top, 1.0, 1000.0);
            }
            else
              // set projection as perspective
              camera->setProjectionMatrixAsPerspective(30.0, aspect_ratio, 1.0, 1000.0);
            break;
          }
        }
      }
    }
    traverse(node, nv);
  }

private:
  bool m_orthographic = false;
};

int main() {
  // TODO: scroll event is has no response with FlatManipulator and Projection_callback

  // change the default TrackballManipulator to our FlatManipulator
  auto key_switch = new osgGA::KeySwitchMatrixManipulator;
  key_switch->addMatrixManipulator('1', "Trackball", new osgGA::TrackballManipulator);
  key_switch->addMatrixManipulator('2', "Flat", new cm::FlatManipulator);
  cm::get_global_viewer().composite_viewer().getView(0)->setCameraManipulator(key_switch);

  // add orthographic projection switch callback
  cm::get_global_viewer().composite_viewer().getView(0)->getCamera()->addEventCallback(
    new Projection_callback);
  cm::get_global_viewer().run();

  // export OSG_FILE_PATH
  cm::get_global_viewer().add_node(osgDB::readNodeFile("lz.osg"));
}
