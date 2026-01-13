#ifndef CM_FLATMANIPULATOR_H
#define CM_FLATMANIPULATOR_H

#include <osgGA/StandardManipulator>

namespace cm {

/*!
 * \brief This class is a two-dimension manipulator, which can only view, pan,
 * and scale (but not rotate) the scene as if it is projected onto the XOY plane.
 * \sa osgGA::OrbitManipulator
 */
class FlatManipulator : public osgGA::StandardManipulator {
public:
  /*!
   * \brief Constructor.
   */
  FlatManipulator() : _distance(1.0) {
    setMinimumDistance(0.05, true);
    setWheelZoomFactor(0.1);
    if(_flags & SET_CENTER_ON_WHEEL_FORWARD_MOVEMENT)
      setAnimationTime(0.2);
  }

  /*!
   * \brief Get the current position and the attitude matrix of this manipulator.
   * \return manipulator matrix
   */
  virtual osg::Matrixd getMatrix() const {
    osg::Matrixd matrix;
    matrix.makeTranslate(0.0f, 0.0f, _distance);
    matrix.postMultTranslate(_center);
    return matrix;
  }

  /*!
   * \brief Get the matrix of the camera manipulator and inverse it.
   * The inverted matrix is typically treated as the view matrix of the camera.
   * \return inverted manipulator matrix
   */
  virtual osg::Matrixd getInverseMatrix() const {
    osg::Matrixd matrix;
    matrix.makeTranslate(0.0f, 0.0f, -_distance);
    matrix.preMultTranslate(-_center);
    return matrix;
  }

  /*!
   * \brief Set up the position matrix of the manipulator.
   * Can be called from user-level code.
   */
  virtual void setByMatrix(const osg::Matrixd &matrix) {
    setByInverseMatrix(osg::Matrixd::inverse(matrix));
  }

  /*!
   * \brief Set up the position matrix of the manipulator.
   * Can be called from user-level code.
   */
  virtual void setByInverseMatrix(const osg::Matrixd &matrix) {
    osg::Vec3d eye, center, up;
    matrix.getLookAt(eye, center, up);
    _center = center;
    _center.z() = 0.0f;
    if (_node.valid())
      _distance = std::abs((_node->getBound().center() - eye).z());
    else
      _distance = std::abs((eye - center).length());
  }

  // Leave empty as we don't need these here. They are used by
  // other functions and classes to set up the manipulator directly.
  virtual void setTransformation(const osg::Vec3d &, const osg::Quat &) {}
  virtual void setTransformation(const osg::Vec3d &, const osg::Vec3d &, const osg::Vec3d &) {}
  virtual void getTransformation(osg::Vec3d &, osg::Quat &) const {}
  virtual void getTransformation(osg::Vec3d &, osg::Vec3d &, osg::Vec3d &) const {}

  /*!
   * \brief Move the camera to its home position.
   * We compute the suitable home position directly and the default is ignored.
   */
  virtual void home(double) {
    if (_node.valid()) {
      _center = _node->getBound().center();
      _center.z() = 0.0f;
      _distance = 3.0 * _node->getBound().radius();
    }
    else {
      _center.set(osg::Vec3());
      _distance = 1.0;
    }
  }

  /*!
   * \brief Move the camera to its home position.
   * We compute the suitable home position directly and the default is ignored.
   */
  virtual void home(const osgGA::GUIEventAdapter &ea, osgGA::GUIActionAdapter &us) {
    home(ea.getTime());
  }

  /** Set the mouse wheel zoom factor.
      The amount of camera movement on each mouse wheel event
      is computed as the current distance to the center multiplied by this factor.
      For example, value of 0.1 will short distance to center by 10% on each wheel up event.
      Use negative value for reverse mouse wheel direction.*/
  void setWheelZoomFactor(double wheelZoomFactor) {
    _wheelZoomFactor = wheelZoomFactor;
  }

  /** Set the minimum distance of the eye point from the center
      before the center is pushed forward.*/
  void setMinimumDistance(const double &minimumDistance, bool relativeToModelSize) {
    _minimumDistance = minimumDistance;
    setRelativeFlag(_minimumDistanceFlagIndex, relativeToModelSize);
  }

protected:
  /*!
   * \brief Destructor.
   */
  virtual ~FlatManipulator() {}

  /*!
   * \brief Zoom the camera when the mouse wheel is scrolled.
   * \param ea osgGA::GUIEventAdapter
   * \param us osgGA::GUIActionAdapter
   * \return true if the event is handled, false otherwise
   */
  virtual bool handleMouseWheel(const osgGA::GUIEventAdapter &ea, osgGA::GUIActionAdapter &us) {
    osgGA::GUIEventAdapter::ScrollingMotion sm = ea.getScrollingMotion();

    // handle centering
    if (_flags & SET_CENTER_ON_WHEEL_FORWARD_MOVEMENT) {
      if (((sm == osgGA::GUIEventAdapter::SCROLL_DOWN && _wheelZoomFactor > 0.)) ||
        ((sm == osgGA::GUIEventAdapter::SCROLL_UP && _wheelZoomFactor < 0.))) {
        if (getAnimationTime() <= 0.) {
          // center by mouse intersection (no animation)
          setCenterByMousePointerIntersection(ea, us);
        }
        else {
          // start new animation only if there is no animation in progress
          if (!isAnimating())
            startAnimationByMousePointerIntersection(ea, us);
        }
      }
    }

    switch (sm) {
      // mouse scroll up event
      case osgGA::GUIEventAdapter::SCROLL_UP: {
        // perform zoom
        zoomModel(_wheelZoomFactor, true);
        us.requestRedraw();
        us.requestContinuousUpdate(isAnimating() || _thrown);
        return true;
      }

      // mouse scroll down event
      case osgGA::GUIEventAdapter::SCROLL_DOWN: {
        // perform zoom
        zoomModel(-_wheelZoomFactor, true);
        us.requestRedraw();
        us.requestContinuousUpdate(isAnimating() || _thrown);
        return true;
      }

      // unhandled mouse scrolling motion
      default:
        return false;
    }
  }

  /*!
   * \brief Pan the camera when the mouse is dragged with left button.
   * \param eventTimeDelta event time delta
   * \param dx mouse movement along x axis
   * \param dy mouse movement along y axis
   * \return true if the movement is handled, false otherwise
   */
  virtual bool performMovementLeftMouseButton(
    const double eventTimeDelta, const double dx, const double dy) {
    // pan model
    float scale = -0.3f * _distance * getThrowScale(eventTimeDelta);
    panModel(dx * scale, dy * scale);

    return true;
  }

  /** Moves camera in x,y,z directions given in camera local coordinates.*/
  virtual void panModel(const float dx, const float dy, const float dz = 0.0f) {
    _center += osg::Vec3d(dx, dy, dz);
  }

  /** Changes the distance of camera to the focal center.
      If pushForwardIfNeeded is true and minimumDistance is reached,
      the focal center is moved forward. Otherwise, distance is limited
      to its minimum value.
      \sa FlatManipulator::setMinimumDistance
  */
  virtual void zoomModel(const float dy, bool pushForwardIfNeeded) {
    // scale
    float scale = 1.0f + dy;

    // minimum distance
    float minDist = _minimumDistance;
    if (getRelativeFlag(_minimumDistanceFlagIndex))
      minDist *= _modelSize;

    if (_distance * scale > minDist) {
      // regular zoom
      _distance *= scale;
    }
    else  {
      if (pushForwardIfNeeded) {
        // push the camera forward
        float yscale = -_distance;
        // osg::Matrixd rotation_matrix(_rotation);
        // osg::Vec3d dv = (osg::Vec3d(0.0f, 0.0f, -1.0f) * rotation_matrix) * (dy * yscale);
        osg::Vec3d dv = osg::Vec3d(0.0f, 0.0f, -1.0f) * (dy * yscale);
        _center += dv;
      }
      else {
        // set distance on its minimum value
        _distance = minDist;
      }
    }
  }

  osg::Vec3 _center;
  double _distance;
  double _wheelZoomFactor;
  double _minimumDistance;

  static int _minimumDistanceFlagIndex;
};

int FlatManipulator::_minimumDistanceFlagIndex =
  osgGA::StandardManipulator::allocateRelativeFlag();

} // namespace cm

#endif // CM_FLATMANIPULATOR_H
