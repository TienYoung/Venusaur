#pragma once

#include <glm/glm.hpp>
#include <glm/ext.hpp>

#define WORLDUP glm::vec3(0.0f, 1.0f, 0.0f)


class Camera
{
public:
	Camera();

	Camera(const glm::vec3& origin, float vfov, float aspect, float aperture, float focal_length);

	static const glm::vec3& m_defaultForward;

	void UVWFrame(glm::vec3& U, glm::vec3& V, glm::vec3& W);

	inline float GetLensRadius() const { return m_aperture * 0.5f; }

	inline const glm::vec3& GetPosition() const { return m_position; }

	void SetForward(glm::vec3 direction);

	inline void MoveForward(float speed) { m_position += m_forward * speed; }

	inline void MoveRight(float speed) { m_position += normalize(m_u) * speed; }

	inline void MoveUp(float speed) { m_position += normalize(m_v) * speed; }

	//void pitch(float speed)
	//{
	//	glm::quat rotation = glm::rotate(glm::quat(1, 0, 0, 0), glm::radians(speed), u);
	//	m_target = m_position + normalize(-w * rotation);
	//	update_uvw();
	//}

	//void yaw(float speed)
	//{
	//	glm::quat rotation = glm::rotate(glm::quat(1, 0, 0, 0), glm::radians(speed), v);
	//	m_target = m_position + normalize(-w * rotation);
	//	update_uvw();
	//}

	//void roll(float speed)
	//{
	//	glm::quat rotation = glm::rotate(glm::quat(1, 0, 0, 0), glm::radians(speed), w);
	//	m_target = m_position + normalize(-w * rotation);
	//	update_uvw();
	//}

private:
	glm::vec3 m_position = glm::vec3(0.0f);
	glm::vec3 m_forward  = m_defaultForward;
	float m_vfov         = 45.0f;
	float m_aspect       = 1.6f;
	float m_aperture     = 0.0f;
	float m_focalLength  = 1.0f;

	glm::vec3 m_u, m_v, m_w;

	void UpdateUVW();
};
