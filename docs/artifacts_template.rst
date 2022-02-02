#{{ jobid }}: {{ job_name }}
============================

* Job ID: {{ jobid }}
* Run ID: {{ runid }}
* Date: {{ job_date }}
* Description: {{ job_description }}



{% if job_artifacts %}

Artifacts
---------

{% for a in job_artifacts -%}
* `{{ a.name }} <{{ a.url }}>`_
{% endfor %}

{% endif %}




{% if job_log %}

Log
---

.. list-table:: Frozen Delights!
    :header-rows: 1
    :widths: 10 30

    * - Timestamp
      - Message
    {% for message in job_log -%}
    * - {{ message.timestamp }}
      - {{ message.msg }}
    {% endfor %}

{% endif %}
