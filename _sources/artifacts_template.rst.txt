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

**Log**

{{ job_log }}

{% endif %}
