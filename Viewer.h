#ifndef CM_VIEWER_H
#define CM_VIEWER_H

#include <vector>
#include <queue>
#include <mutex>
#include <thread>

#include <osg/Group>
#include <osg/Callback>
#include <osgViewer/CompositeViewer>
#include <osgViewer/config/SingleWindow>
#include <osgGA/TrackballManipulator>

namespace osg {

// Xterm system colors
// https://jonasjacek.github.io/colors/

static const Vec4f BLACK(0.0f, 0.0f, 0.0f, 1.0f);
static const Vec4f MAROON(0.5f, 0.0f, 0.0f, 1.0f);
static const Vec4f GREEN(0.0f, 0.5f, 0.0f, 1.0f);
static const Vec4f OLIVE(0.5f, 0.5f, 0.0f, 1.0f);
static const Vec4f NAVY(0.0f, 0.0f, 0.5f, 1.0f);
static const Vec4f PURPLE(0.5f, 0.0f, 0.5f, 1.0f);
static const Vec4f TEAL(0.0f, 0.5f, 0.5f, 1.0f);
static const Vec4f SILVER(0.75f, 0.75f, 0.75f, 1.0f);
static const Vec4f GREY(0.5f, 0.5f, 0.5f, 1.0f);
static const Vec4f RED(1.0f, 0.0f, 0.0f, 1.0f);
static const Vec4f LIME(0.0f, 1.0f, 0.0f, 1.0f);
static const Vec4f YELLOW(1.0f, 1.0f, 0.0f, 1.0f);
static const Vec4f BLUE(0.0f, 0.0f, 1.0f, 1.0f);
static const Vec4f FUCHSIA(1.0f, 0.0f, 1.0f, 1.0f);
static const Vec4f AQUA(0.0f, 1.0f, 1.0f, 1.0f);
static const Vec4f WHITE(1.0f, 1.0f, 1.0f, 1.0f);

}

namespace cm {

/*!
 * \brief Simple composite viewer.
 * It add new nodes to the root group node and rendered in a thread safe way.
 */
class Viewer {
  /*!
   * \brief Add node callback functor for each view.
   */
  struct Add_node_callback : public osg::NodeCallback {
    /*!
     * \brief Constructs from the node queue.
     * \param m reference to the queue mutex
     * \param q reference to the queue
     */
    Add_node_callback(std::mutex &m, std::queue<osg::Node *> &q) :
      m_mutex(m), m_queue(q) {}

    /*!
     * \brief Update callback entry function.
     * \param node callback node
     * \param nv node visitor
     */
    virtual void operator()(osg::Node *node, osg::NodeVisitor *nv) {
      auto group = static_cast<osg::Group *>(node);
      if (group) {
        std::lock_guard<std::mutex> lock(m_mutex);
        // add all queued node to the graph
        while (!m_queue.empty()) {
          group->addChild(m_queue.front());
          m_queue.pop();
        }
      }
      traverse(node, nv);
    }

    std::mutex &m_mutex;
    std::queue<osg::Node *> &m_queue;
  };

public:
  /*!
   * \brief Constructor, allocate data and initialize the viewer.
   * \param row number of rows of sub view
   * \param column number of columns of sub view
   * \param width width of each sub view
   * \param height height of each sub view
   */
  Viewer(
    const std::size_t row = 1,
    const std::size_t column = 1,
    const std::size_t width = 1600,
    const std::size_t height = 900) :
    m_row(row),
    m_column(column),
    m_width(width),
    m_height(height),
    m_mutexes(row * column),
    m_queues(row * column) {
    // default TrackballManipulator with auto home position
    auto manipulator = new osgGA::TrackballManipulator;
    for (std::size_t i = 0; i < m_row; ++i) {
      for (std::size_t j = 0; j < m_column; ++j) {
        // sub view
        auto view = new osgViewer::View;
        // allocate root node and set call back for each view
        auto group = new osg::Group;
        group->addUpdateCallback(
          new Add_node_callback(m_mutexes[i * m_column + j], m_queues[i * m_column + j]));
        view->setSceneData(group);
        view->getCamera()->setClearColor(osg::WHITE);
        // view port origin is at the lower left corner
        // relative to the graphics context window
        view->getCamera()->setViewport(
          j * m_width, (m_row - 1 - i) * m_height, m_width, m_height);
        view->getCamera()->setProjectionMatrixAsPerspective(
          30.0, double(m_width) / m_height, 1.0, 1000.0);
        // use shared manipulator
        view->setCameraManipulator(manipulator);
        // add view
        m_viewer.addView(view);
      }
    }
  };

  /*!
   * \brief Destructor, joins the rendering thead when destructed.
   */
  ~Viewer() {
    m_thread.join();
  }

  /*!
   * \brief Deleted copy constructor.
   */
  Viewer(const Viewer &) = delete;

  /*!
   * \brief Deleted copy assignment operator.
   */
  Viewer &operator=(const Viewer &) = delete;

  /*!
   * \brief Thread entry functor, create context and run the viewer in the same thread.
   * \note When this rendering thread finishes, its graphics context is closed.
   * \param screen_num screen number
   */
  void operator()(const int screen_num) {
    // create graphics context
    auto traits = new osg::GraphicsContext::Traits;
    traits->x = 50;
    traits->y = 50;
    traits->width = static_cast<int>(m_column * m_width);
    traits->height = static_cast<int>(m_row * m_height);
    traits->windowName = "View Manager";
    traits->windowDecoration = true;
    traits->samples = 4;
    traits->doubleBuffer = true;
    traits->sharedContext = 0;
    traits->screenNum = screen_num;
    auto gc = osg::GraphicsContext::createGraphicsContext(traits);
    gc->setClearColor(osg::BLACK);
    gc->setClearMask(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // sharing graphics context
    for (unsigned int i = 0; i < m_viewer.getNumViews(); ++i)
      m_viewer.getView(i)->getCamera()->setGraphicsContext(gc);

    // run the renderig loop
    m_viewer.run();

    // close the graphics context in the same thread
    gc->close();
  }

  /*!
   * \brief Start the rendering thread.
   * \param screen_num screen number
   */
  void run(const int screen_num = 1) {
    m_thread = std::thread(std::ref(*this), screen_num);
  }

  /*!
   * \brief Move the manipulator to the default home position.
   * You can also move to home position by pressing the space bar.
   */
  void move_to_home_position() {
    // inject space key into the EventQueue
    auto view = m_viewer.getView(0);
    view->getEventQueue()->keyPress(' ');
  }

  /*!
   * \brief Adds node to the root node queue.
   * \param n node pointer
   * \param i subview index, must in [0, row * column)
   */
  void add_node(osg::Node *n, const std::size_t i = 0) {
    std::lock_guard<std::mutex> lock(m_mutexes[i]);
    m_queues[i].push(n);
  }

  /*!
   * \brief Adds node to the root node queue.
   * \param n node pointer
   * \param r subview row, must in [0, row)
   * \param c subview column, must in [0, column)
   */
  void add_node(osg::Node *n, const std::size_t r, const std::size_t c) {
    add_node(n, r * m_column + c);
  }

  /*!
   * \brief Access subview root node.
   * \param i subview index, must in [0, row * column)
   * \return subview root node
   */
  osg::Group *root(const std::size_t i = 0) {
    return m_viewer.getView(i)->getSceneData()->asGroup();
  }

  /*!
   * \brief Access subview root node.
   * \param r subview row, must in [0, row)
   * \param c subview column, must in [0, column)
   * \return subview root node
   */
  osg::Group *root(const std::size_t r, const std::size_t c) {
    return root(r * m_column + c);
  }

  /*!
   * \brief Access subview const root node.
   * \param i subview index, must in [0, row * column)
   * \return subview root node
   */
  const osg::Group *root(const std::size_t i = 0) const {
    return m_viewer.getView(i)->getSceneData()->asGroup();
  }

  /*!
   * \brief Access subview const root node.
   * \param r subview row, must in [0, row)
   * \param c subview column, must in [0, column)
   * \return subview root node
   */
  const osg::Group *root(const std::size_t r, const std::size_t c) const {
    return root(r * m_column + c);
  }

  /*!
   * \brief Access the composite viewer.
   * \return composite viewer
   */
  osgViewer::CompositeViewer &composite_viewer() {
    return m_viewer;
  }

  /*!
   * \brief Access the const composite viewer.
   * \return composite viewer
   */
  const osgViewer::CompositeViewer &composite_viewer() const {
    return m_viewer;
  }

private:
  // number of view rows
  const std::size_t m_row;
  // number of view columns
  const std::size_t m_column;
  // width of each view
  const std::size_t m_width;
  // height of each view
  const std::size_t m_height;
  // the underlying composite viewer
  osgViewer::CompositeViewer m_viewer;
  // mutexes for each queue
  std::vector<std::mutex> m_mutexes;
  // node queue
  std::vector<std::queue<osg::Node *>> m_queues;
  // rendering thread
  std::thread m_thread;
};

} // namespace cm

#endif // CM_VIEWER_H
